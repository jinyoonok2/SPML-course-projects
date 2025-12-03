import java.io.*;
import java.nio.file.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.stream.Collectors;

/**
 * FeatureEngine - Handles all feature extraction and engineering
 * 
 * Responsibilities:
 * - Extract features from time-series data using sliding windows
 * - Calculate statistical features (mean, std, median, RMS)
 * - Support different window sizes and sliding parameters
 * - Feature normalization and selection
 * 
 * Consolidated from Part1, Part2, and Part3 feature extraction logic
 */
public class FeatureEngine {
    
    /**
     * Extract basic 6 features (mean, std per axis) using sliding windows
     * Based on Part1_DataProcessor.extractFeatures() and Part2 window logic
     */
    public static void extractBasicFeatures(Path baseDir, Path outputCsv, int windowMs, int slideMs) 
            throws IOException {
        Path formattedDir = baseDir.resolve("formatted_data");
        
        if (!Files.isDirectory(formattedDir)) {
            throw new IOException("formatted_data directory not found. Run data formatting first.");
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
                    String label = DataManager.extractActivityLabel(file.getFileName().toString());
                    int windows = processFileForBasicFeatures(file, label, bw, windowMs, slideMs);
                    fileCount++;
                    windowCount += windows;
                    System.out.println("[features] " + file.getFileName() + " -> " + windows + " windows");
                }
            }
            
            System.out.println("[DONE] Processed " + fileCount + " files, generated " + windowCount + " feature windows");
        }
    }
    
    /**
     * Extract expanded 12 features (mean, std, median, RMS per axis)
     * Extended version for Part3 feature expansion
     */
    public static void extractExpandedFeatures(Path baseDir, Path outputCsv, int windowMs, int slideMs) 
            throws IOException {
        Path formattedDir = baseDir.resolve("formatted_data");
        
        try (BufferedWriter bw = Files.newBufferedWriter(outputCsv, StandardCharsets.UTF_8,
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {

            // Write CSV header for 12 features
            bw.write("mean_x,std_x,mean_y,std_y,mean_z,std_z,median_x,median_y,median_z,rms_x,rms_y,rms_z,Activity");
            bw.newLine();

            int fileCount = 0;
            int windowCount = 0;

            try (DirectoryStream<Path> ds = Files.newDirectoryStream(formattedDir, "*.csv")) {
                for (Path file : ds) {
                    String label = DataManager.extractActivityLabel(file.getFileName().toString());
                    int windows = processFileForExpandedFeatures(file, label, bw, windowMs, slideMs);
                    fileCount++;
                    windowCount += windows;
                    System.out.println("[features] " + file.getFileName() + " -> " + windows + " windows (12 features)");
                }
            }
            
            System.out.println("[DONE] Processed " + fileCount + " files, generated " + windowCount + " expanded feature windows");
        }
    }
    
    /**
     * Process single file for basic 6 features
     * Extracted from Part1_DataProcessor.processFileToFeatures()
     */
    private static int processFileForBasicFeatures(Path file, String label, BufferedWriter bw, int windowMs, int slideMs) 
            throws IOException {
        List<DataPoint> dataPoints = readDataPoints(file);
        if (dataPoints.isEmpty()) return 0;

        int windowCount = 0;
        long startTime = dataPoints.get(0).timestamp;
        long endTime = dataPoints.get(dataPoints.size() - 1).timestamp;

        for (long windowStart = startTime; windowStart < endTime; windowStart += slideMs) {
            long windowEnd = windowStart + windowMs;

            List<DataPoint> windowData = extractWindowData(dataPoints, windowStart, windowEnd);
            if (windowData.size() < 10) continue; // Skip windows with too few samples

            // Calculate 6 features
            double[] features = calculateBasicFeatures(windowData);

            // Write to CSV
            bw.write(String.format("%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%s",
                    features[0], features[1], features[2], features[3], features[4], features[5], label));
            bw.newLine();
            windowCount++;
        }

        return windowCount;
    }
    
    /**
     * Process single file for expanded 12 features
     */
    private static int processFileForExpandedFeatures(Path file, String label, BufferedWriter bw, int windowMs, int slideMs) 
            throws IOException {
        List<DataPoint> dataPoints = readDataPoints(file);
        if (dataPoints.isEmpty()) return 0;

        int windowCount = 0;
        long startTime = dataPoints.get(0).timestamp;
        long endTime = dataPoints.get(dataPoints.size() - 1).timestamp;

        for (long windowStart = startTime; windowStart < endTime; windowStart += slideMs) {
            long windowEnd = windowStart + windowMs;

            List<DataPoint> windowData = extractWindowData(dataPoints, windowStart, windowEnd);
            if (windowData.size() < 10) continue;

            // Calculate 12 features
            double[] features = calculateExpandedFeatures(windowData);

            // Write to CSV
            bw.write(String.format("%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%s",
                    features[0], features[1], features[2], features[3], features[4], features[5],
                    features[6], features[7], features[8], features[9], features[10], features[11], label));
            bw.newLine();
            windowCount++;
        }

        return windowCount;
    }
    
    /**
     * Read all data points from formatted file
     */
    private static List<DataPoint> readDataPoints(Path file) throws IOException {
        List<DataPoint> dataPoints = new ArrayList<>();
        
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
        
        return dataPoints;
    }
    
    /**
     * Extract data points within time window
     */
    private static List<DataPoint> extractWindowData(List<DataPoint> allData, long windowStart, long windowEnd) {
        List<DataPoint> windowData = new ArrayList<>();
        for (DataPoint dp : allData) {
            if (dp.timestamp >= windowStart && dp.timestamp < windowEnd) {
                windowData.add(dp);
            }
        }
        return windowData;
    }
    
    /**
     * Calculate basic 6 features: mean and std for each axis
     * Based on Part1_DataProcessor.calculateFeatures()
     */
    private static double[] calculateBasicFeatures(List<DataPoint> data) {
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
     * Calculate expanded 12 features: basic + median + RMS
     * Extended for Part3 feature expansion
     */
    private static double[] calculateExpandedFeatures(List<DataPoint> data) {
        // Get basic 6 features
        double[] basic = calculateBasicFeatures(data);
        
        // Calculate additional features
        double[] xValues = data.stream().mapToDouble(p -> p.ax).sorted().toArray();
        double[] yValues = data.stream().mapToDouble(p -> p.ay).sorted().toArray();
        double[] zValues = data.stream().mapToDouble(p -> p.az).sorted().toArray();
        
        double medianX = calculateMedian(xValues);
        double medianY = calculateMedian(yValues);
        double medianZ = calculateMedian(zValues);
        
        double rmsX = Math.sqrt(data.stream().mapToDouble(p -> p.ax * p.ax).average().orElse(0));
        double rmsY = Math.sqrt(data.stream().mapToDouble(p -> p.ay * p.ay).average().orElse(0));
        double rmsZ = Math.sqrt(data.stream().mapToDouble(p -> p.az * p.az).average().orElse(0));
        
        // Combine all 12 features
        return new double[]{basic[0], basic[1], basic[2], basic[3], basic[4], basic[5],
                           medianX, medianY, medianZ, rmsX, rmsY, rmsZ};
    }
    
    /**
     * Calculate median of sorted array
     */
    private static double calculateMedian(double[] sorted) {
        int n = sorted.length;
        if (n % 2 == 0) {
            return (sorted[n/2-1] + sorted[n/2]) / 2.0;
        } else {
            return sorted[n/2];
        }
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