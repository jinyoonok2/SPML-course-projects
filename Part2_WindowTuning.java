import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import weka.core.*;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;

/**
 * Part2_WindowTuning
 * ------------------
 * Part 2: Time window tuning (1s, 2s, 3s, 4s with 1s sliding window)
 * 
 * This class:
 *   1. Tests different window sizes: 1s, 2s, 3s, 4s
 *   2. Uses 1-second sliding window (stride) for 2s, 3s, 4s
 *   3. Extracts same 6 features: mean_x, std_x, mean_y, std_y, mean_z, std_z
 *   4. Trains Decision Tree (J48) classifier for each window size
 *   5. Reports accuracy for each to find optimal window size
 */
public class Part2_WindowTuning {

    private static final int SLIDE_MS = 1000; // 1 second sliding window
    private static final int[] WINDOW_SIZES_MS = {1000, 2000, 3000, 4000}; // 1s, 2s, 3s, 4s

    public static void run() throws Exception {
        Path baseDir = Paths.get(".").toAbsolutePath().normalize();
        Path formattedDataDir = baseDir.resolve("formatted_data");

        if (!Files.isDirectory(formattedDataDir)) {
            throw new IOException("formatted_data directory not found. Please run Part 1 first.");
        }

        System.out.println("Testing window sizes: 1s, 2s, 3s, 4s (with 1s sliding)");
        System.out.println();

        double[] accuracies = new double[WINDOW_SIZES_MS.length];

        for (int i = 0; i < WINDOW_SIZES_MS.length; i++) {
            int windowMs = WINDOW_SIZES_MS[i];
            double windowSec = windowMs / 1000.0;

            System.out.println("=" .repeat(80));
            System.out.println("Testing Window Size: " + windowSec + " seconds");
            System.out.println("=" .repeat(80));

            Path featuresCsv = baseDir.resolve("features_part2_" + windowSec + "s.csv");
            Path featuresArff = baseDir.resolve("features_part2_" + windowSec + "s.arff");

            // Extract features with current window size
            System.out.println("Extracting features with " + windowSec + "s windows...");
            extractFeatures(formattedDataDir, featuresCsv, windowMs, SLIDE_MS);
            System.out.println("✓ Features saved to: " + featuresCsv);
            System.out.println();

            // Convert to ARFF
            System.out.println("Converting to ARFF format...");
            convertToArff(featuresCsv, featuresArff);
            System.out.println("✓ ARFF file saved to: " + featuresArff);
            System.out.println();

            // Train and evaluate
            System.out.println("Training Decision Tree classifier...");
            double accuracy = evaluateClassifier(featuresArff);
            accuracies[i] = accuracy;
            System.out.println();
        }

        // Summary
        System.out.println("=" .repeat(80));
        System.out.println("SUMMARY: Window Size Comparison");
        System.out.println("=" .repeat(80));
        System.out.println();
        System.out.printf("%-15s %10s%n", "Window Size", "Accuracy");
        System.out.println("-".repeat(30));

        int bestIndex = 0;
        for (int i = 0; i < WINDOW_SIZES_MS.length; i++) {
            double windowSec = WINDOW_SIZES_MS[i] / 1000.0;
            System.out.printf("%-15s %9.2f%%%n", windowSec + "s", accuracies[i]);
            if (accuracies[i] > accuracies[bestIndex]) {
                bestIndex = i;
            }
        }

        System.out.println("-".repeat(30));
        double bestWindowSec = WINDOW_SIZES_MS[bestIndex] / 1000.0;
        System.out.printf("Best: %.1fs with %.2f%% accuracy%n", bestWindowSec, accuracies[bestIndex]);
        System.out.println();
    }

    /**
     * Extract features with configurable window size and sliding window
     */
    private static void extractFeatures(Path formattedDir, Path outputCsv, int windowMs, int slideMs) throws IOException {
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
                    int windows = processFileToFeatures(file, label, bw, windowMs, slideMs);
                    fileCount++;
                    windowCount += windows;
                }
            }
            System.out.println("  Processed " + fileCount + " files");
            System.out.println("  Generated " + windowCount + " feature windows");
        }
    }

    private static int processFileToFeatures(Path file, String label, BufferedWriter bw, 
                                            int windowMs, int slideMs) throws IOException {
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

        // Create sliding windows
        long startTime = dataPoints.get(0).timestamp;
        long endTime = dataPoints.get(dataPoints.size() - 1).timestamp;
        int windowCount = 0;

        for (long windowStart = startTime; windowStart + windowMs <= endTime; windowStart += slideMs) {
            long windowEnd = windowStart + windowMs;

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

            // Class attribute
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
    private static double evaluateClassifier(Path arffFile) throws Exception {
        // Load data
        DataSource source = new DataSource(arffFile.toString());
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);

        System.out.println("  Dataset: " + data.numInstances() + " instances, " + 
                          data.numAttributes() + " attributes");

        // Build classifier
        J48 tree = new J48();
        tree.buildClassifier(data);

        // Evaluate with 10-fold cross-validation
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(tree, data, 10, new Random(1));

        // Print results
        System.out.println("  Accuracy: " + String.format("%.2f%%", eval.pctCorrect()));
        System.out.println("  Kappa: " + String.format("%.4f", eval.kappa()));

        return eval.pctCorrect();
    }

    /**
     * Extract activity label from filename
     */
    private static String extractLabelFromFilename(String filename) {
        String[] parts = filename.split("-");
        if (parts.length < 5) return "unknown";

        int activityIndex = 4;
        String activity = parts[activityIndex].toLowerCase();

        // Check for non_hand_wash FIRST to avoid substring matching issue
        if (activity.contains("non_hand_wash") || activity.contains("no_hand_wash") || activity.contains("not_hand_wash")) 
            return "non_hand_wash";
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
