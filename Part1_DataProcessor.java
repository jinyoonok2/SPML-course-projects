import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import weka.core.*;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;

/**
 * Part1_DataProcessor
 * -------------------
 * Part 1: Process all raw data and establish baseline accuracy
 * 
 * This class:
 *   1. Reads all CSV files from raw_data/
 *   2. Formats data (drops first line, keeps timestamp, ax, ay, az)
 *   3. Extracts features using 1-second windows
 *   4. Generates 6 features: mean_x, std_x, mean_y, std_y, mean_z, std_z
 *   5. Converts to ARFF format for Weka
 *   6. Trains Decision Tree (J48) classifier
 *   7. Reports accuracy using 10-fold cross-validation
 */
public class Part1_DataProcessor {

    private static final int WINDOW_SIZE_MS = 1000; // 1 second window

    public static void run() throws Exception {
        Path baseDir = Paths.get(".").toAbsolutePath().normalize();
        Path rawDataDir = baseDir.resolve("raw_data");
        Path formattedDataDir = baseDir.resolve("formatted_data");
        Path featuresCsv = baseDir.resolve("features_part1.csv");
        Path featuresArff = baseDir.resolve("features_part1.arff");

        // Step 1: Format raw data
        System.out.println("Step 1: Formatting raw data...");
        formatRawData(rawDataDir, formattedDataDir);
        System.out.println("✓ Formatted data saved to: " + formattedDataDir);
        System.out.println();

        // Step 2: Extract features
        System.out.println("Step 2: Extracting features (1-second windows)...");
        extractFeatures(formattedDataDir, featuresCsv);
        System.out.println("✓ Features saved to: " + featuresCsv);
        System.out.println();

        // Step 3: Convert to ARFF
        System.out.println("Step 3: Converting to ARFF format...");
        convertToArff(featuresCsv, featuresArff);
        System.out.println("✓ ARFF file saved to: " + featuresArff);
        System.out.println();

        // Step 4: Train and evaluate classifier
        System.out.println("Step 4: Training Decision Tree classifier...");
        evaluateClassifier(featuresArff);
    }

    /**
     * Format raw CSV files: drop first line, keep timestamp, ax, ay, az
     */
    private static void formatRawData(Path rawDir, Path formattedDir) throws IOException {
        if (!Files.isDirectory(rawDir)) {
            throw new IOException("raw_data directory not found: " + rawDir);
        }
        Files.createDirectories(formattedDir);

        int fileCount = 0;
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(rawDir, "*.csv")) {
            for (Path src : ds) {
                Path dst = formattedDir.resolve(src.getFileName());
                formatFile(src, dst);
                fileCount++;
            }
        }
        System.out.println("  Formatted " + fileCount + " files");
    }

    private static void formatFile(Path src, Path dst) throws IOException {
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
     * Extract features from formatted data using 1-second windows
     */
    private static void extractFeatures(Path formattedDir, Path outputCsv) throws IOException {
        if (!Files.isDirectory(formattedDir)) {
            throw new IOException("formatted_data directory not found: " + formattedDir);
        }

        try (BufferedWriter bw = Files.newBufferedWriter(outputCsv, StandardCharsets.UTF_8,
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {

            // Write CSV header
            bw.write("mean_x,std_x,mean_y,std_y,mean_z,std_z,Activity");
            bw.newLine();

            int fileCount = 0;
            int windowCount = 0;

            try (DirectoryStream<Path> ds = Files.newDirectoryStream(formattedDir, "*.csv")) {
                for (Path file : ds) {
                    String label = extractLabelFromFilename(file.getFileName().toString());
                    int windows = processFileToFeatures(file, label, bw);
                    fileCount++;
                    windowCount += windows;
                }
            }
            System.out.println("  Processed " + fileCount + " files");
            System.out.println("  Generated " + windowCount + " feature windows");
        }
    }

    private static int processFileToFeatures(Path file, String label, BufferedWriter bw) throws IOException {
        List<DataPoint> dataPoints = new ArrayList<>();

        // Read all data points
        try (BufferedReader br = Files.newBufferedReader(file, StandardCharsets.UTF_8)) {
            for (String line; (line = br.readLine()) != null; ) {
                line = line.trim();
                if (line.isEmpty()) continue;

                String[] tokens = line.split(",");
                if (tokens.length != 4) continue;

                long timestamp = Long.parseLong(tokens[0].trim());
                double ax = Double.parseDouble(tokens[1].trim());
                double ay = Double.parseDouble(tokens[2].trim());
                double az = Double.parseDouble(tokens[3].trim());

                dataPoints.add(new DataPoint(timestamp, ax, ay, az));
            }
        }

        if (dataPoints.isEmpty()) return 0;

        // Create windows
        long startTime = dataPoints.get(0).timestamp;
        long endTime = dataPoints.get(dataPoints.size() - 1).timestamp;
        int windowCount = 0;

        for (long windowStart = startTime; windowStart < endTime; windowStart += WINDOW_SIZE_MS) {
            long windowEnd = windowStart + WINDOW_SIZE_MS;

            List<DataPoint> windowData = new ArrayList<>();
            for (DataPoint dp : dataPoints) {
                if (dp.timestamp >= windowStart && dp.timestamp < windowEnd) {
                    windowData.add(dp);
                }
            }

            if (windowData.size() < 10) continue; // Skip windows with too few samples

            // Calculate features
            double[] features = calculateFeatures(windowData);

            // Write to CSV
            bw.write(String.format("%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%s",
                    features[0], features[1], features[2], features[3], features[4], features[5], label));
            bw.newLine();
            windowCount++;
        }

        return windowCount;
    }

    /**
     * Calculate 6 features: mean and std for each axis
     */
    private static double[] calculateFeatures(List<DataPoint> data) {
        int n = data.size();
        double sumX = 0, sumY = 0, sumZ = 0;
        double sumX2 = 0, sumY2 = 0, sumZ2 = 0;

        for (DataPoint dp : data) {
            sumX += dp.ax;
            sumY += dp.ay;
            sumZ += dp.az;
            sumX2 += dp.ax * dp.ax;
            sumY2 += dp.ay * dp.ay;
            sumZ2 += dp.az * dp.az;
        }

        double meanX = sumX / n;
        double meanY = sumY / n;
        double meanZ = sumZ / n;

        double varX = (sumX2 / n) - (meanX * meanX);
        double varY = (sumY2 / n) - (meanY * meanY);
        double varZ = (sumZ2 / n) - (meanZ * meanZ);

        double stdX = Math.sqrt(Math.max(0, varX));
        double stdY = Math.sqrt(Math.max(0, varY));
        double stdZ = Math.sqrt(Math.max(0, varZ));

        return new double[]{meanX, stdX, meanY, stdY, meanZ, stdZ};
    }

    /**
     * Convert CSV to ARFF format for Weka
     */
    private static void convertToArff(Path csvFile, Path arffFile) throws IOException {
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

        System.out.println("  Generated " + data.size() + " instances");
    }

    /**
     * Train Decision Tree and evaluate with 10-fold cross-validation
     */
    private static void evaluateClassifier(Path arffFile) throws Exception {
        // Load data
        DataSource source = new DataSource(arffFile.toString());
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);

        System.out.println("  Dataset: " + data.numInstances() + " instances, " + 
                          data.numAttributes() + " attributes");
        System.out.println("  Classes: " + data.numClasses());
        for (int i = 0; i < data.numClasses(); i++) {
            System.out.println("    - " + data.classAttribute().value(i));
        }
        System.out.println();

        // Build classifier
        J48 tree = new J48();
        tree.buildClassifier(data);

        // Evaluate with 10-fold cross-validation
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(tree, data, 10, new Random(1));

        // Print results
        System.out.println("=== Decision Tree (J48) Results ===");
        System.out.println();
        System.out.println("Accuracy: " + String.format("%.2f%%", eval.pctCorrect()));
        System.out.println("Kappa: " + String.format("%.4f", eval.kappa()));
        System.out.println();
        System.out.println("=== Confusion Matrix ===");
        double[][] matrix = eval.confusionMatrix();
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                System.out.print((int)matrix[i][j] + " ");
            }
            System.out.println();
        }
        System.out.println();
        System.out.println("=== Detailed Statistics ===");
        System.out.println(eval.toClassDetailsString());
    }

    /**
     * Extract activity label from filename
     * Format: WatchID-AssignmentX-Subject-Hand-Activity-Info-DateTime.csv
     */
    private static String extractLabelFromFilename(String filename) {
        String[] parts = filename.split("-");
        if (parts.length < 5) return "unknown";

        // Find activity field (usually index 4 when AssignmentX is present)
        int activityIndex = 4;
        String activity = parts[activityIndex].toLowerCase();

        // Check for non_hand_wash FIRST (before hand_wash) to avoid substring matching issue
        if (activity.contains("non_hand_wash") || activity.contains("no_hand_wash") || activity.contains("not_hand_wash")) return "non_hand_wash";
        if (activity.contains("hand_wash")) return "hand_wash";

        return activity;
    }

    /**
     * Data point holder class
     */
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
}
