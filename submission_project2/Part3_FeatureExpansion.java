import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import weka.core.*;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;

/**
 * Part3_FeatureExpansion
 * ----------------------
 * Part 3: Add median and RMS features (12 features total)
 * 
 * This class:
 *   1. Reads best window size from Part 2 config file
 *   2. Extracts 12 features: mean, std, median, RMS per axis
 *   3. Uses the optimal window size determined in Part 2
 *   4. Trains Decision Tree (J48) classifier
 *   5. Compares accuracy with Part 2 (6 features) vs Part 3 (12 features)
 */
public class Part3_FeatureExpansion {

    private static final int SLIDE_MS = 1000; // 1 second sliding window

    public static void run() throws Exception {
        Path baseDir = Paths.get(".").toAbsolutePath().normalize();
        Path formattedDataDir = baseDir.resolve("formatted_data");

        if (!Files.isDirectory(formattedDataDir)) {
            throw new IOException("formatted_data directory not found. Please run Part 1 first.");
        }

        // Get best window size from Part 2
        int bestWindowMs = Part2_WindowTuning.getBestWindowSize(baseDir);
        double bestWindowSec = bestWindowMs / 1000.0;

        System.out.println("Using optimal window size from Part 2: " + bestWindowSec + "s");
        System.out.println("Expanding features from 6 to 12 (adding median and RMS)");
        System.out.println();

        // Create results folder for Part 3
        Path part3Dir = baseDir.resolve("results/part3");
        Files.createDirectories(part3Dir);

        Path featuresCsv = part3Dir.resolve("features.csv");
        Path featuresArff = part3Dir.resolve("features.arff");

        // Extract features with 12 features
        System.out.println("Step 1: Extracting 12 features...");
        System.out.println("  Features: mean, std, median, RMS for each axis (x, y, z)");
        extractFeatures(formattedDataDir, featuresCsv, bestWindowMs, SLIDE_MS);
        System.out.println("✓ Features saved to: " + featuresCsv);
        System.out.println();

        // Convert to ARFF
        System.out.println("Step 2: Converting to ARFF format...");
        convertToArff(featuresCsv, featuresArff);
        System.out.println("✓ ARFF file saved to: " + featuresArff);
        System.out.println();

        // Train and evaluate
        System.out.println("Step 3: Training Decision Tree classifier...");
        Path resultsFile = part3Dir.resolve("results.txt");
        evaluateClassifier(featuresArff, resultsFile, bestWindowSec);
        
        System.out.println();
        System.out.println("=" .repeat(80));
        System.out.println("Compare with Part 2 results (6 features) to see improvement!");
        System.out.println("=" .repeat(80));
    }

    /**
     * Extract 12 features: mean, std, median, RMS for each axis
     */
    private static void extractFeatures(Path formattedDir, Path outputCsv, int windowMs, int slideMs) throws IOException {
        try (BufferedWriter bw = Files.newBufferedWriter(outputCsv, StandardCharsets.UTF_8,
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {

            // Write CSV header with 12 features
            bw.write("mean_x,std_x,median_x,rms_x,mean_y,std_y,median_y,rms_y,mean_z,std_z,median_z,rms_z,Activity");
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

            // Calculate 12 features
            double[] features = calculateFeatures(windowData);

            // Write to CSV
            bw.write(String.format("%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%s",
                    features[0], features[1], features[2], features[3],   // x: mean, std, median, rms
                    features[4], features[5], features[6], features[7],   // y: mean, std, median, rms
                    features[8], features[9], features[10], features[11], // z: mean, std, median, rms
                    label));
            bw.newLine();
            windowCount++;
        }

        return windowCount;
    }

    /**
     * Calculate 12 features: mean, std, median, RMS for each axis
     */
    private static double[] calculateFeatures(List<DataPoint> data) {
        int n = data.size();
        
        // Separate data by axis
        double[] xData = new double[n];
        double[] yData = new double[n];
        double[] zData = new double[n];
        
        for (int i = 0; i < n; i++) {
            xData[i] = data.get(i).ax;
            yData[i] = data.get(i).ay;
            zData[i] = data.get(i).az;
        }
        
        // Calculate features for each axis
        double[] xFeatures = calculateAxisFeatures(xData);
        double[] yFeatures = calculateAxisFeatures(yData);
        double[] zFeatures = calculateAxisFeatures(zData);
        
        // Combine: mean, std, median, rms for x, y, z
        return new double[]{
            xFeatures[0], xFeatures[1], xFeatures[2], xFeatures[3],  // x
            yFeatures[0], yFeatures[1], yFeatures[2], yFeatures[3],  // y
            zFeatures[0], zFeatures[1], zFeatures[2], zFeatures[3]   // z
        };
    }

    /**
     * Calculate 4 features for a single axis: mean, std, median, RMS
     */
    private static double[] calculateAxisFeatures(double[] data) {
        int n = data.length;
        
        // Mean
        double sum = 0;
        for (double v : data) {
            sum += v;
        }
        double mean = sum / n;
        
        // Standard deviation
        double sumSq = 0;
        for (double v : data) {
            sumSq += v * v;
        }
        double variance = (sumSq / n) - (mean * mean);
        double std = Math.sqrt(Math.max(0, variance));
        
        // Median
        double[] sorted = data.clone();
        Arrays.sort(sorted);
        double median;
        if (n % 2 == 0) {
            median = (sorted[n/2 - 1] + sorted[n/2]) / 2.0;
        } else {
            median = sorted[n/2];
        }
        
        // RMS (Root Mean Square)
        double rms = Math.sqrt(sumSq / n);
        
        return new double[]{mean, std, median, rms};
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
    private static void evaluateClassifier(Path arffFile, Path resultsFile, double windowSec) throws Exception {
        // Load data
        DataSource source = new DataSource(arffFile.toString());
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);

        System.out.println("  Dataset: " + data.numInstances() + " instances, " + 
                          data.numAttributes() + " attributes (12 features)");
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

        // Create output string
        StringBuilder output = new StringBuilder();
        output.append("================================================================================\n");
        output.append("PART 3: Feature Expansion Results\n");
        output.append("================================================================================\n\n");
        output.append("Configuration:\n");
        output.append("  - Window Size: ").append(windowSec).append(" seconds (from Part 2)\n");
        output.append("  - Features: 12 (mean, std, median, RMS per axis)\n");
        output.append("  - Classifier: Decision Tree (J48)\n");
        output.append("  - Validation: 10-fold cross-validation\n\n");
        
        output.append("Dataset Information:\n");
        output.append("  - Instances: ").append(data.numInstances()).append("\n");
        output.append("  - Attributes: ").append(data.numAttributes()).append(" (12 features)\n");
        output.append("  - Classes: ").append(data.numClasses()).append("\n");
        for (int i = 0; i < data.numClasses(); i++) {
            output.append("    - ").append(data.classAttribute().value(i)).append("\n");
        }
        output.append("\n");

        // Print results to console
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

        // Append results to output
        output.append("=== Decision Tree (J48) Results ===\n\n");
        output.append("Accuracy: ").append(String.format("%.2f%%", eval.pctCorrect())).append("\n");
        output.append("Kappa: ").append(String.format("%.4f", eval.kappa())).append("\n\n");
        
        output.append("=== Confusion Matrix ===\n");
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                output.append((int)matrix[i][j]).append(" ");
            }
            output.append("\n");
        }
        output.append("\n");
        
        output.append("=== Detailed Statistics ===\n");
        output.append(eval.toClassDetailsString());
        
        // Save to file
        try (BufferedWriter bw = Files.newBufferedWriter(resultsFile, StandardCharsets.UTF_8,
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {
            bw.write(output.toString());
        }
        
        System.out.println();
        System.out.println("✓ Results saved to: " + resultsFile);
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
