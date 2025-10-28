import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

/**
 * WadaManager
 * -----------
 * Fixed workflow with no optional arguments.
 *
 * Commands (run from the subject folder containing raw_data/):
 *   1) format   : read raw_data/*.csv, drop first line, keep (timestamp,ax,ay,az) -> formatted_data/
 *   2) features : read formatted_data/*.csv, build 1s-window features (mean/std per axis) -> features.csv
 */
public class WadaManager {

    public static void main(String[] args) {
        if (args.length < 1) {
            System.err.println("Subcommands: format | features");
            System.exit(2);
        }
        String cmd = args[0];
        Path baseDir = Paths.get(".").toAbsolutePath().normalize();
        try {
            switch (cmd) {
                case "format":
                    format(baseDir);
                    break;
                case "features":
                    Path out = baseDir.resolve("features.csv");
                    features(baseDir, out);
                    break;
                default:
                    System.err.println("Unknown subcommand: " + cmd);
                    System.exit(2);
            }
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    // ---------------- step 1: format (raw_data -> formatted_data) ----------------

    /**
     * Reads raw CSVs from ./raw_data:
     *  - Drops the first line (artifact) per file.
     *  - Parses comma- or whitespace-separated rows.
     *  - If 6 cols: (timestamp_ms, sensor_type, accuracy, ax, ay, az) -> keep sensor_type==1, then write 4 cols
     *  - If 4 cols: (timestamp_ms, ax, ay, az) -> write as-is (4 cols)
     * Writes cleaned CSVs to ./formatted_data with same filenames.
     */
    private static void format(Path dir) throws IOException {
        Path in = dir.resolve("raw_data");
        Path out = dir.resolve("formatted_data");
        if (!Files.isDirectory(in)) {
            throw new IOException("raw_data not found: " + in);
        }
        Files.createDirectories(out);

        try (DirectoryStream<Path> ds = Files.newDirectoryStream(in, "*.csv")) {
            for (Path src : ds) {
                Path dst = out.resolve(src.getFileName().toString());
                filterUsefulColumnsDroppingFirst(src, dst);
                System.out.println("[format] raw_data/" + src.getFileName() + " -> formatted_data/" + dst.getFileName());
            }
        }
    }

    private static void filterUsefulColumnsDroppingFirst(Path src, Path dst) throws IOException {
        try (BufferedReader br = Files.newBufferedReader(src, StandardCharsets.UTF_8);
             BufferedWriter bw = Files.newBufferedWriter(dst, StandardCharsets.UTF_8,
                     StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {

            // Drop the very first line (often a partial row/artifact)
            br.readLine();

            for (String line; (line = br.readLine()) != null; ) {
                line = line.trim();
                if (line.isEmpty()) continue;

                String[] toks = splitSmart(line);
                if (toks.length == 6) {
                    long ts = toLong(toks[0]);
                    int sensorType = toInt(toks[1], -1);
                    if (sensorType != 1) continue; // accelerometer only
                    double ax = toDouble(toks[3]);
                    double ay = toDouble(toks[4]);
                    double az = toDouble(toks[5]);
                    bw.write(Long.toString(ts) + "," + d(ax) + "," + d(ay) + "," + d(az));
                    bw.newLine();
                } else if (toks.length == 4) {
                    long ts = toLong(toks[0]);
                    double ax = toDouble(toks[1]);
                    double ay = toDouble(toks[2]);
                    double az = toDouble(toks[3]);
                    bw.write(Long.toString(ts) + "," + d(ax) + "," + d(ay) + "," + d(az));
                    bw.newLine();
                } else {
                    // ignore unexpected rows quietly
                }
            }
        }
    }

    // ---------------- step 2: features (formatted_data -> features.csv) ----------------

    private static void features(Path dir, Path outCsv) throws IOException {
        Path in = dir.resolve("formatted_data");
        if (!Files.isDirectory(in)) {
            throw new IOException("formatted_data not found: " + in);
        }

        try (BufferedWriter bw = Files.newBufferedWriter(outCsv, StandardCharsets.UTF_8,
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {
            bw.write("mean_x,std_x,mean_y,std_y,mean_z,std_z,Activity");
            bw.newLine();

            try (DirectoryStream<Path> ds = Files.newDirectoryStream(in, "*.csv")) {
                for (Path f : ds) {
                    String label = labelFromFilename(f.getFileName().toString());
                    processOneFileToFeatures(f, label, bw);
                    System.out.println("[features] " + f.getFileName() + " -> appended");
                }
            }
        }
        System.out.println("[DONE] features.csv -> " + outCsv);
    }

    private static void processOneFileToFeatures(Path file, String label, BufferedWriter bw) throws IOException {
        Map<Long, Stats> bucket = new HashMap<>();
        long t0 = -1L;
        long maxBucket = -1L;

        try (BufferedReader br = Files.newBufferedReader(file, StandardCharsets.UTF_8)) {
            for (String line; (line = br.readLine()) != null; ) {
                line = line.trim();
                if (line.isEmpty()) continue;
                String[] toks = line.split(",", -1);
                if (toks.length != 4) continue; // formatted_data must be 4 cols

                long ts = toLong(toks[0]);
                double ax = toDouble(toks[1]);
                double ay = toDouble(toks[2]);
                double az = toDouble(toks[3]);

                if (t0 < 0) t0 = ts;
                long b = (ts - t0) / 1000L;
                maxBucket = Math.max(maxBucket, b);
                bucket.computeIfAbsent(b, k -> new Stats()).add(ax, ay, az);
            }
        }

        if (maxBucket <= 0) return; // nothing or too short

        long lastFull = maxBucket - 1; // ignore last (possibly incomplete) window
        for (long b = 0; b <= lastFull; b++) {
            Stats s = bucket.get(b);
            if (s == null || s.n == 0) continue;
            double[] feat = s.meanStd();
            bw.write(d(feat[0]) + "," + d(feat[1]) + "," + d(feat[2]) + "," + d(feat[3]) + "," + d(feat[4]) + "," + d(feat[5]) + "," + label);
            bw.newLine();
        }
    }

    // ---------------- helpers ----------------

    private static String[] splitSmart(String line) {
        if (line.indexOf(',') >= 0) return line.split(",", -1);
        return line.trim().split("\\s+", -1);
    }

    private static long toLong(String s) {
        try {
            if (s.indexOf('E') >= 0 || s.indexOf('e') >= 0) {
                return (long)Math.round(Double.parseDouble(s));
            }
            return Long.parseLong(s.trim());
        } catch (NumberFormatException e) {
            return (long)Math.round(Double.parseDouble(s.trim()));
        }
    }

    private static int toInt(String s, int def) {
        try { return Integer.parseInt(s.trim()); } catch (Exception e) { return def; }
    }

    private static double toDouble(String s) {
        return Double.parseDouble(s.trim());
    }

    private static String d(double v) {
        return Double.toString(v); // compact decimal (no scientific format)
    }

    private static String labelFromFilename(String name) {
        // watchID-AssignmentX-subject-hand-activity-info-YYYY-...csv
        String[] p = name.split("-");
        if (p.length < 6) return "unknown";

        int idxActivity;
        if (p[1].toLowerCase(Locale.ROOT).startsWith("assignment")) {
            // With AssignmentX token, activity is at index 4
            idxActivity = 4;
        } else {
            // Without AssignmentX, activity was at index 3
            idxActivity = 3;
        }
        if (idxActivity >= p.length) return "unknown";

        String a = p[idxActivity].toLowerCase(Locale.ROOT);
        if (a.equals("hand_wash")) return "hand_wash";
        if (a.equals("non_hand_wash") || a.equals("no_hand_wash") || a.equals("not_hand_wash"))
            return "non_hand_wash"; // normalize to one spelling
        return a;
    }

    private static class Stats {
        long n = 0;
        double sx=0, sy=0, sz=0;
        double sxx=0, syy=0, szz=0;
        void add(double x, double y, double z) {
            n++;
            sx += x; sy += y; sz += z;
            sxx += x*x; syy += y*y; szz += z*z;
        }
        double[] meanStd() {
            double mx = sx/n, my = sy/n, mz = sz/n;
            double vx = sxx/n - mx*mx;
            double vy = syy/n - my*my;
            double vz = szz/n - mz*mz;
            if (vx < 0) vx = 0;
            if (vy < 0) vy = 0;
            if (vz < 0) vz = 0;
            return new double[]{ mx, Math.sqrt(vx), my, Math.sqrt(vy), mz, Math.sqrt(vz) };
        }
    }
}

// -----------------------------------------------------------------------------
// # 1) Format: drop first line & keep timestamp,ax,ay,az -> formatted_data/
// java WadaManager format
//
// # 2) Features: build 1s-window features -> features.csv
// java WadaManager features
// -----------------------------------------------------------------------------
