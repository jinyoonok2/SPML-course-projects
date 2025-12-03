import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.stream.Collectors;

/**
 * WadaManager
 * -----------
 * Unified workflow manager for all assignment parts.
 *
 * Commands (run from the subject folder containing raw_data/):
 *   1) format     : read raw_data/*.csv, drop first line, keep (timestamp,ax,ay,az) -> formatted_data/
 *   2) features   : read formatted_data/*.csv, build 1s-window features (mean/std per axis) -> features.csv
 *   3) part1      : format + features + train J48 classifier (baseline)
 *   4) part2      : test multiple window sizes (1s, 2s, 3s, 4s) + find best
 *   5) part3      : extract 12 features using best window + train J48
 */
public class WadaManager {

    public static void main(String[] args) {
        if (args.length < 1) {
            System.err.println("Subcommands: format | features | part1 | part2 | part3 | part4 | part5 | all");
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
                case "part1":
                    runPart1(baseDir);
                    break;
                case "part2":
                    runPart2(baseDir);
                    break;
                case "part3":
                    runPart3(baseDir);
                    break;
                case "part4":
                    runPart4(baseDir);
                    break;
                case "part5":
                    runPart5(baseDir);
                    break;
                case "all":
                    runAll(baseDir);
                    break;
                default:
                    System.err.println("Unknown subcommand: " + cmd);
                    System.exit(2);
            }
        } catch (Exception e) {
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

    // ---------------- Part 1: Baseline (format + features + classify) ----------------
    
    private static void runPart1(Path baseDir) throws Exception {
        System.out.println("=== Part 1: Baseline Accuracy ===");
        System.out.println("Processing all raw data with Decision Tree classifier");
        System.out.println("Window: 1 second, Features: 6 (mean, std per axis)");
        System.out.println();

        // Create results directory
        Path part1Dir = baseDir.resolve("results/part1");
        Files.createDirectories(part1Dir);

        // Step 1: Format data
        System.out.println("Step 1: Formatting raw data...");
        format(baseDir);

        // Step 2: Extract features (6 features, 1s window)
        System.out.println("Step 2: Extracting features (1-second windows)...");
        Path featuresCsv = part1Dir.resolve("features.csv");
        features(baseDir, featuresCsv);

        // Step 3: Convert to ARFF and train classifier
        System.out.println("Step 3: Training Decision Tree classifier...");
        Path featuresArff = part1Dir.resolve("features.arff");
        MyWekaUtils.csvToArff(featuresCsv.toString(), featuresArff.toString());
        
        double accuracy = MyWekaUtils.trainAndEvaluate(featuresArff.toString(), 1); // J48
        System.out.printf("✓ Baseline accuracy (J48): %.4f%n", accuracy);
        
        System.out.println("Part 1 completed successfully!");
    }

    // ---------------- Part 2: Window tuning ----------------
    
    private static void runPart2(Path baseDir) throws Exception {
        System.out.println("=== Part 2: Window Size Tuning ===");
        System.out.println("Testing window sizes: 1s, 2s, 3s, 4s (with 1s sliding)");
        System.out.println();

        Path formattedDataDir = baseDir.resolve("formatted_data");
        if (!Files.isDirectory(formattedDataDir)) {
            System.out.println("Formatted data not found. Running format step...");
            format(baseDir);
        }

        Path part2Dir = baseDir.resolve("results/part2");
        Files.createDirectories(part2Dir);

        int[] windowSizes = {1000, 2000, 3000, 4000}; // ms
        double bestAccuracy = 0;
        int bestWindowMs = 1000;
        
        StringBuilder results = new StringBuilder();
        results.append("Window Size (s),Accuracy\n");

        for (int windowMs : windowSizes) {
            double windowSec = windowMs / 1000.0;
            System.out.printf("Testing %.0fs window...\n", windowSec);
            
            Path featuresCsv = part2Dir.resolve("features_" + (int)windowSec + "s.csv");
            Path featuresArff = part2Dir.resolve("features_" + (int)windowSec + "s.arff");
            
            // Extract features with specific window size
            extractFeaturesWithWindow(formattedDataDir, featuresCsv, windowMs, 1000); // 1s slide
            
            // Train and evaluate
            MyWekaUtils.csvToArff(featuresCsv.toString(), featuresArff.toString());
            double accuracy = MyWekaUtils.trainAndEvaluate(featuresArff.toString(), 1); // J48
            
            System.out.printf("✓ %.0fs window accuracy: %.4f%n", windowSec, accuracy);
            results.append(String.format("%.0f,%.4f%n", windowSec, accuracy));
            
            if (accuracy > bestAccuracy) {
                bestAccuracy = accuracy;
                bestWindowMs = windowMs;
            }
        }
        
        // Save results and best config
        Files.write(part2Dir.resolve("window_tuning_results.csv"), results.toString().getBytes());
        Files.write(part2Dir.resolve("best_config.txt"), 
                   String.format("best_window_ms=%d\nbest_accuracy=%.4f", bestWindowMs, bestAccuracy).getBytes());
        
        System.out.printf("\n✓ Best window: %.0fs (accuracy: %.4f)\n", bestWindowMs/1000.0, bestAccuracy);
        System.out.println("Part 2 completed successfully!");
    }

    // ---------------- Part 3: Feature expansion ----------------
    
    private static void runPart3(Path baseDir) throws Exception {
        System.out.println("=== Part 3: Feature Expansion ===");
        System.out.println("Adding median and RMS features (12 features total)");
        
        // Read best window size from Part 2
        int bestWindowMs = getBestWindowSize(baseDir);
        System.out.printf("Using optimal window size: %.0fs\n", bestWindowMs/1000.0);
        System.out.println();

        Path formattedDataDir = baseDir.resolve("formatted_data");
        if (!Files.isDirectory(formattedDataDir)) {
            System.out.println("Formatted data not found. Running format step...");
            format(baseDir);
        }

        Path part3Dir = baseDir.resolve("results/part3");
        Files.createDirectories(part3Dir);

        Path featuresCsv = part3Dir.resolve("features.csv");
        Path featuresArff = part3Dir.resolve("features.arff");

        // Extract 12 features
        System.out.println("Step 1: Extracting 12 features (mean, std, median, RMS per axis)...");
        extractExpandedFeatures(formattedDataDir, featuresCsv, bestWindowMs, 1000);

        // Train and evaluate
        System.out.println("Step 2: Training Decision Tree with 12 features...");
        MyWekaUtils.csvToArff(featuresCsv.toString(), featuresArff.toString());
        double accuracy = MyWekaUtils.trainAndEvaluate(featuresArff.toString(), 1); // J48
        
        System.out.printf("✓ 12-feature accuracy (J48): %.4f%n", accuracy);
        System.out.println("Part 3 completed successfully!");
    }

    // ---------------- Part 4: Feature Selection ----------------
    
    private static void runPart4(Path baseDir) throws Exception {
        System.out.println("=== Part 4: Sequential Feature Selection ===");
        System.out.println("Finding optimal feature subset using forward selection");
        System.out.println("Features: 12 available features, Classifier: Decision Tree");
        System.out.println();

        Path part3FeaturesPath = baseDir.resolve("results/part3/features.csv");
        if (!Files.exists(part3FeaturesPath)) {
            System.err.println("Part 3 features not found. Please run Part 3 first.");
            return;
        }

        Path part4Dir = baseDir.resolve("results/part4");
        Files.createDirectories(part4Dir);

        // Read CSV data
        String[][] csvData = MyWekaUtils.readCSV(part3FeaturesPath.toString());
        if (csvData == null || csvData.length == 0) {
            System.err.println("Features CSV is empty.");
            return;
        }

        // Run Sequential Feature Selection for Decision Tree
        SFSResult result = runSequentialFeatureSelection(csvData, 1); // 1 = Decision Tree
        
        // Save results
        saveSFSResults(part4Dir.resolve("dt_sfs_curve.csv"), result);
        
        System.out.printf("✓ Best feature subset (%d features): %s%n", 
                         result.selected.size(), result.selected);
        System.out.printf("✓ Final accuracy: %.4f%n", result.finalAcc);
        System.out.println("Part 4 completed successfully!");
    }

    // ---------------- Part 5: Classifier Comparison ----------------
    
    private static void runPart5(Path baseDir) throws Exception {
        System.out.println("=== Part 5: Classifier Comparison ===");
        System.out.println("Comparing Decision Tree, Random Forest, and SVM with feature selection");
        System.out.println();

        Path part3FeaturesPath = baseDir.resolve("results/part3/features.csv");
        if (!Files.exists(part3FeaturesPath)) {
            System.err.println("Part 3 features not found. Please run Part 3 first.");
            return;
        }

        Path part5Dir = baseDir.resolve("results/part5");
        Files.createDirectories(part5Dir);

        // Read CSV data
        String[][] csvData = MyWekaUtils.readCSV(part3FeaturesPath.toString());
        if (csvData == null || csvData.length == 0) {
            System.err.println("Features CSV is empty.");
            return;
        }

        // Run SFS for all three classifiers
        System.out.println("Running Sequential Feature Selection for Decision Tree...");
        SFSResult dtResult = runSequentialFeatureSelection(csvData, 1); // Decision Tree
        saveSFSResults(part5Dir.resolve("dt_sfs_curve.csv"), dtResult);
        
        System.out.println("Running Sequential Feature Selection for Random Forest...");
        SFSResult rfResult = runSequentialFeatureSelection(csvData, 2); // Random Forest
        saveSFSResults(part5Dir.resolve("rf_sfs_curve.csv"), rfResult);
        
        System.out.println("Running Sequential Feature Selection for SVM...");
        SFSResult svmResult = runSequentialFeatureSelection(csvData, 3); // SVM
        saveSFSResults(part5Dir.resolve("svm_sfs_curve.csv"), svmResult);

        // Print comparison
        System.out.println("\n=== Classifier Comparison Summary ===");
        printClassifierSummary("Decision Tree", dtResult);
        printClassifierSummary("Random Forest", rfResult);
        printClassifierSummary("SVM", svmResult);
        
        System.out.println("Part 5 completed successfully!");
    }

    // ---------------- Enhanced feature extraction methods ----------------
    
    private static void extractFeaturesWithWindow(Path formattedDir, Path outCsv, int windowMs, int slideMs) 
            throws IOException {
        try (BufferedWriter bw = Files.newBufferedWriter(outCsv, StandardCharsets.UTF_8,
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {
            bw.write("mean_x,std_x,mean_y,std_y,mean_z,std_z,Activity");
            bw.newLine();

            try (DirectoryStream<Path> ds = Files.newDirectoryStream(formattedDir, "*.csv")) {
                for (Path f : ds) {
                    String label = labelFromFilename(f.getFileName().toString());
                    processFileWithWindow(f, label, bw, windowMs, slideMs, false);
                }
            }
        }
    }
    
    private static void extractExpandedFeatures(Path formattedDir, Path outCsv, int windowMs, int slideMs) 
            throws IOException {
        try (BufferedWriter bw = Files.newBufferedWriter(outCsv, StandardCharsets.UTF_8,
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {
            bw.write("mean_x,std_x,mean_y,std_y,mean_z,std_z,median_x,median_y,median_z,rms_x,rms_y,rms_z,Activity");
            bw.newLine();

            try (DirectoryStream<Path> ds = Files.newDirectoryStream(formattedDir, "*.csv")) {
                for (Path f : ds) {
                    String label = labelFromFilename(f.getFileName().toString());
                    processFileWithWindow(f, label, bw, windowMs, slideMs, true);
                }
            }
        }
    }

    private static void processFileWithWindow(Path file, String label, BufferedWriter bw, 
            int windowMs, int slideMs, boolean expandedFeatures) throws IOException {
        
        // Read all data points first
        List<DataPoint> points = new ArrayList<>();
        try (BufferedReader br = Files.newBufferedReader(file, StandardCharsets.UTF_8)) {
            for (String line; (line = br.readLine()) != null; ) {
                line = line.trim();
                if (line.isEmpty()) continue;
                String[] toks = line.split(",", -1);
                if (toks.length != 4) continue;
                
                long ts = toLong(toks[0]);
                double ax = toDouble(toks[1]);
                double ay = toDouble(toks[2]);
                double az = toDouble(toks[3]);
                points.add(new DataPoint(ts, ax, ay, az));
            }
        }
        
        if (points.isEmpty()) return;
        
        // Extract windows with sliding
        long startTime = points.get(0).timestamp;
        long endTime = points.get(points.size() - 1).timestamp;
        
        for (long windowStart = startTime; windowStart + windowMs <= endTime; windowStart += slideMs) {
            long windowEnd = windowStart + windowMs;
            
            List<DataPoint> windowPoints = points.stream()
                .filter(p -> p.timestamp >= windowStart && p.timestamp < windowEnd)
                .collect(java.util.stream.Collectors.toList());
                
            if (windowPoints.size() < 10) continue; // Skip windows with too few points
            
            if (expandedFeatures) {
                double[] features = extractAllFeatures(windowPoints);
                bw.write(String.format("%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%s",
                    features[0], features[1], features[2], features[3], features[4], features[5],
                    features[6], features[7], features[8], features[9], features[10], features[11], label));
            } else {
                double[] features = extractBasicFeatures(windowPoints);
                bw.write(String.format("%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%s",
                    features[0], features[1], features[2], features[3], features[4], features[5], label));
            }
            bw.newLine();
        }
    }
    
    private static double[] extractBasicFeatures(List<DataPoint> points) {
        Stats stats = new Stats();
        for (DataPoint p : points) {
            stats.add(p.ax, p.ay, p.az);
        }
        return stats.meanStd();
    }
    
    private static double[] extractAllFeatures(List<DataPoint> points) {
        // Extract basic features (mean, std)
        double[] basic = extractBasicFeatures(points);
        
        // Extract median and RMS
        double[] x = points.stream().mapToDouble(p -> p.ax).sorted().toArray();
        double[] y = points.stream().mapToDouble(p -> p.ay).sorted().toArray();
        double[] z = points.stream().mapToDouble(p -> p.az).sorted().toArray();
        
        double medianX = median(x);
        double medianY = median(y);
        double medianZ = median(z);
        
        double rmsX = Math.sqrt(points.stream().mapToDouble(p -> p.ax * p.ax).average().orElse(0));
        double rmsY = Math.sqrt(points.stream().mapToDouble(p -> p.ay * p.ay).average().orElse(0));
        double rmsZ = Math.sqrt(points.stream().mapToDouble(p -> p.az * p.az).average().orElse(0));
        
        return new double[]{basic[0], basic[1], basic[2], basic[3], basic[4], basic[5],
                           medianX, medianY, medianZ, rmsX, rmsY, rmsZ};
    }
    
    private static double median(double[] sorted) {
        int n = sorted.length;
        if (n % 2 == 0) {
            return (sorted[n/2-1] + sorted[n/2]) / 2.0;
        } else {
            return sorted[n/2];
        }
    }
    
    private static int getBestWindowSize(Path baseDir) throws IOException {
        Path configFile = baseDir.resolve("results/part2/best_config.txt");
        if (!Files.exists(configFile)) {
            System.err.println("Warning: Part 2 config not found, using default 1s window");
            return 1000;
        }
        
        String content = new String(Files.readAllBytes(configFile));
        for (String line : content.split("\n")) {
            if (line.startsWith("best_window_ms=")) {
                return Integer.parseInt(line.substring("best_window_ms=".length()));
            }
        }
        return 1000;
    }
    
    // Helper class for data points
    private static class DataPoint {
        long timestamp;
        double ax, ay, az;
        
        DataPoint(long timestamp, double ax, double ay, double az) {
            this.timestamp = timestamp;
            this.ax = ax;
            this.ay = ay;
            this.az = az;
        }
    }
    
    // Helper class for SFS results
    private static class SFSResult {
        List<Integer> selected = new ArrayList<>();
        List<Double> accCurve = new ArrayList<>();
        double finalAcc = 0.0;
        int iterations = 0;
    }
    
    // Sequential Feature Selection implementation
    private static SFSResult runSequentialFeatureSelection(String[][] csvData, int classifierType) 
            throws Exception {
        
        final int NUM_FEATURES = 12;
        final double MIN_IMPROVEMENT = 0.001; // 0.1%
        
        SFSResult result = new SFSResult();
        Set<Integer> available = new HashSet<>();
        for (int i = 0; i < NUM_FEATURES; i++) {
            available.add(i);
        }
        
        double bestAcc = 0.0;
        
        while (!available.isEmpty()) {
            int bestFeature = -1;
            double bestIterAcc = bestAcc;
            
            // Try adding each available feature
            for (int candidate : available) {
                List<Integer> testSet = new ArrayList<>(result.selected);
                testSet.add(candidate);
                
                double acc = MyWekaUtils.evaluateFeatureSubset(csvData, testSet, classifierType);
                
                if (acc > bestIterAcc) {
                    bestIterAcc = acc;
                    bestFeature = candidate;
                }
            }
            
            // Check if improvement is significant
            if (bestFeature == -1 || (bestIterAcc - bestAcc) < MIN_IMPROVEMENT) {
                break; // No significant improvement
            }
            
            // Add best feature
            result.selected.add(bestFeature);
            available.remove(bestFeature);
            bestAcc = bestIterAcc;
            result.accCurve.add(bestAcc);
            result.iterations++;
            
            System.out.printf("  Iteration %d: Added feature %d, accuracy = %.4f%n", 
                             result.iterations, bestFeature, bestAcc);
        }
        
        result.finalAcc = bestAcc;
        return result;
    }
    
    private static void saveSFSResults(Path outputPath, SFSResult result) throws IOException {
        try (BufferedWriter bw = Files.newBufferedWriter(outputPath, StandardCharsets.UTF_8)) {
            bw.write("Iteration,Accuracy\n");
            for (int i = 0; i < result.accCurve.size(); i++) {
                bw.write(String.format("%d,%.4f\n", i + 1, result.accCurve.get(i)));
            }
        }
    }
    
    private static void printClassifierSummary(String classifierName, SFSResult result) {
        System.out.printf("%-15s: %d features, accuracy = %.4f, iterations = %d%n",
                         classifierName, result.selected.size(), result.finalAcc, result.iterations);
        System.out.printf("                Selected features: %s%n", result.selected);
    }
    
    // ---------------- Run All Parts ----------------
    
    private static void runAll(Path baseDir) throws Exception {
        System.out.println("=" .repeat(80));
        System.out.println("RUNNING ALL PARTS OF ASSIGNMENT 2");
        System.out.println("=" .repeat(80));
        System.out.println();
        
        runPart1(baseDir);
        System.out.println();
        
        runPart2(baseDir);
        System.out.println();
        
        runPart3(baseDir);
        System.out.println();
        
        runPart4(baseDir);
        System.out.println();
        
        runPart5(baseDir);
        System.out.println();
        
        System.out.println("=" .repeat(80));
        System.out.println("ALL PARTS COMPLETED SUCCESSFULLY!");
        System.out.println("=" .repeat(80));
    }
}

// -----------------------------------------------------------------------------
// # 1) Format: drop first line & keep timestamp,ax,ay,az -> formatted_data/
// java WadaManager format
//
// # 2) Features: build 1s-window features -> features.csv
// java WadaManager features
//
// # 3) Part 1: Complete baseline analysis (format + features + J48)
// java WadaManager part1
//
// # 4) Part 2: Window size tuning (1s, 2s, 3s, 4s)  
// java WadaManager part2
//
// # 5) Part 3: Feature expansion (12 features + J48)
// java WadaManager part3
//
// # 6) Part 4: Sequential feature selection (Decision Tree)
// java WadaManager part4
//
// # 7) Part 5: Classifier comparison (DT, RF, SVM with SFS)
// java WadaManager part5
// -----------------------------------------------------------------------------
