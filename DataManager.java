import java.io.*;
import java.nio.file.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * DataManager - Handles all raw data processing and formatting
 * 
 * Responsibilities:
 * - Format raw CSV files (drop header, normalize columns)
 * - Convert between data formats 
 * - Basic data validation and cleaning
 * 
 * Extracted and consolidated from Part1_DataProcessor formatting logic
 */
public class DataManager {
    
    /**
     * Format all raw CSV files from raw_data/ to formatted_data/
     * Drops first line, keeps timestamp,ax,ay,az columns only
     */
    public static void formatRawData(Path baseDir) throws IOException {
        Path rawDir = baseDir.resolve("raw_data");
        Path formattedDir = baseDir.resolve("formatted_data");
        
        if (!Files.isDirectory(rawDir)) {
            throw new IOException("raw_data directory not found: " + rawDir);
        }
        
        Files.createDirectories(formattedDir);
        
        int fileCount = 0;
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(rawDir, "*.csv")) {
            for (Path src : ds) {
                Path dst = formattedDir.resolve(src.getFileName());
                formatSingleFile(src, dst);
                System.out.println("[format] " + src.getFileName() + " -> " + dst.getFileName());
                fileCount++;
            }
        }
        
        System.out.println("[DONE] Formatted " + fileCount + " files");
    }
    
    /**
     * Format individual CSV file - extracted from Part1 logic
     */
    private static void formatSingleFile(Path src, Path dst) throws IOException {
        try (BufferedReader br = Files.newBufferedReader(src, StandardCharsets.UTF_8);
             BufferedWriter bw = Files.newBufferedWriter(dst, StandardCharsets.UTF_8,
                     StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {

            // Drop first line (artifact)
            br.readLine();

            for (String line; (line = br.readLine()) != null; ) {
                line = line.trim();
                if (line.isEmpty()) continue;

                String[] tokens = line.split(",");
                if (tokens.length == 6) {
                    // Format: timestamp, sensor_type, accuracy, ax, ay, az
                    int sensorType = Integer.parseInt(tokens[1].trim());
                    if (sensorType != 1) continue; // Skip non-accelerometer data

                    String timestamp = tokens[0].trim();
                    String ax = tokens[3].trim();
                    String ay = tokens[4].trim();
                    String az = tokens[5].trim();
                    bw.write(timestamp + "," + ax + "," + ay + "," + az);
                    bw.newLine();
                } else if (tokens.length == 4) {
                    // Already formatted: timestamp, ax, ay, az
                    bw.write(line);
                    bw.newLine();
                }
            }
        }
    }
    
    /**
     * Convert CSV to ARFF format for Weka processing
     * Extracted from Part1 convertToArff logic
     */
    public static void convertCsvToArff(Path csvFile, Path arffFile) throws IOException {
        List<String[]> data = new ArrayList<>();
        String[] header = null;

        // Read CSV
        try (BufferedReader br = Files.newBufferedReader(csvFile, StandardCharsets.UTF_8)) {
            header = br.readLine().split(",");
            for (String line; (line = br.readLine()) != null; ) {
                data.add(line.split(","));
            }
        }

        // Write ARFF
        try (BufferedWriter bw = Files.newBufferedWriter(arffFile, StandardCharsets.UTF_8,
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {

            bw.write("@RELATION gestures\n\n");

            // Attributes (all numeric except last one which is class)
            for (int i = 0; i < header.length - 1; i++) {
                bw.write("@ATTRIBUTE " + header[i] + " NUMERIC\n");
            }

            // Class attribute - collect unique labels
            Set<String> labels = new TreeSet<>();
            for (String[] row : data) {
                labels.add(row[row.length - 1]);
            }
            bw.write("@ATTRIBUTE Activity {" + String.join(",", labels) + "}\n\n");

            // Data
            bw.write("@DATA\n");
            for (String[] row : data) {
                bw.write(String.join(",", row) + "\n");
            }
        }

        System.out.println("[convert] Generated " + data.size() + " instances -> " + arffFile.getFileName());
    }
    
    /**
     * Extract activity label from filename
     * Based on Part1_DataProcessor.extractLabelFromFilename()
     */
    public static String extractActivityLabel(String filename) {
        String[] parts = filename.split("-");
        if (parts.length < 5) return "unknown";

        // Find activity field (usually index 4 when AssignmentX is present)
        int activityIndex = 4;
        String activity = parts[activityIndex].toLowerCase();

        // Check for non_hand_wash FIRST (before hand_wash) to avoid substring matching issue
        if (activity.contains("non_hand_wash") || activity.contains("no_hand_wash") || activity.contains("not_hand_wash")) {
            return "non_hand_wash";
        }
        if (activity.contains("hand_wash")) {
            return "hand_wash";
        }

        return activity;
    }
}