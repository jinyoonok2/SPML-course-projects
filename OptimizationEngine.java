import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * OptimizationEngine - Parameter optimization and feature selection
 * 
 * Responsibilities:
 * - Window size optimization (testing 1s, 2s, 3s, 4s windows)
 * - Sequential Feature Selection (SFS) for multiple classifiers
 * - Hyperparameter tuning and performance tracking
 * - Result persistence and configuration management
 * 
 * Consolidated from Part2 window tuning and Part4 feature selection logic
 */
public class OptimizationEngine {
    
    private static final int[] WINDOW_SIZES_MS = {1000, 2000, 3000, 4000};
    private static final int SLIDE_MS = 1000;
    private static final double MIN_IMPROVEMENT = 0.001; // 0.1% minimum improvement for SFS
    
    /**
     * Find optimal window size for EACH classifier using basic 6 features (Part 2)
     * Tests 1s, 2s, 3s, 4s windows for J48, RandomForest, and SVM
     * Returns optimal window for each classifier
     */
    public static WindowOptimizationResult findOptimalWindowSize(Path baseDir) throws Exception {
        Path resultsDir = baseDir.resolve("results/2_window_optimization");
        Files.createDirectories(resultsDir);
        
        // Start logging
        ExperimentLogger logger = new ExperimentLogger();
        logger.startLogging(resultsDir.resolve("window_optimization_log.txt"));
        
        try {
            System.out.println("Testing window sizes for each classifier with 6 basic features: 1s, 2s, 3s, 4s");
            System.out.println("Finding optimal window for each classifier independently");
            System.out.println();
            
            // Extract features for each window size once (reused by all classifiers)
        Map<Integer, Path> windowFeatures = new LinkedHashMap<>();
        for (int windowMs : WINDOW_SIZES_MS) {
            Path featuresCsv = resultsDir.resolve(String.format("window_%dms_features.csv", windowMs));
            FeatureEngine.extractBasicFeatures(baseDir, featuresCsv, windowMs, SLIDE_MS);
            windowFeatures.put(windowMs, featuresCsv);
        }
        
        // Store results per classifier
        Map<MLEngine.ClassifierType, Integer> optimalWindows = new LinkedHashMap<>();
        Map<MLEngine.ClassifierType, Double> optimalAccuracies = new LinkedHashMap<>();
        
        // Test each classifier with all window sizes
        for (MLEngine.ClassifierType classifier : MLEngine.ClassifierType.values()) {
            System.out.printf("%n=== Testing %s ===%n", classifier.displayName);
            
            int bestWindowMs = WINDOW_SIZES_MS[0];
            double bestAccuracy = 0.0;
            
            for (int windowMs : WINDOW_SIZES_MS) {
                double windowSec = windowMs / 1000.0;
                Path featuresCsv = windowFeatures.get(windowMs);
                
                double accuracy = MLEngine.evaluateClassifier(featuresCsv, classifier);
                
                System.out.printf("  %.0fs window: %.4f%n", windowSec, accuracy);
                
                if (accuracy > bestAccuracy) {
                    bestAccuracy = accuracy;
                    bestWindowMs = windowMs;
                }
            }
            
            optimalWindows.put(classifier, bestWindowMs);
            optimalAccuracies.put(classifier, bestAccuracy);
            
            // Generate detailed report for best window configuration
            System.out.printf("%n  Generating detailed report for optimal window (%.0fs)...%n", bestWindowMs/1000.0);
            Path bestFeatureCsv = windowFeatures.get(bestWindowMs);
            Path classifierReportDir = resultsDir.resolve(classifier.name().toLowerCase());
            Files.createDirectories(classifierReportDir);
            String experimentName = String.format("WindowOpt_%s_%.0fs", classifier.name(), bestWindowMs/1000.0);
            MLEngine.evaluateClassifier(bestFeatureCsv, classifier, classifierReportDir, experimentName);
            
            System.out.printf("  ✓ Best for %s: %.0fs (accuracy: %.4f)%n", 
                             classifier.displayName, bestWindowMs/1000.0, bestAccuracy);
        }
        
        // Print summary
        System.out.println("=== Window Optimization Summary ===");
        for (MLEngine.ClassifierType classifier : MLEngine.ClassifierType.values()) {
            int optimalMs = optimalWindows.get(classifier);
            double optimalAcc = optimalAccuracies.get(classifier);
            System.out.printf("%-20s: %.0fs window (accuracy: %.4f)%n", 
                             classifier.displayName, optimalMs/1000.0, optimalAcc);
        }
        
        // Save results
        saveOptimalWindowSizes(resultsDir, optimalWindows, optimalAccuracies);
        
        return new WindowOptimizationResult(optimalWindows, optimalAccuracies);
        } finally {
            logger.stopLogging();
        }
    }
    
    /**
     * Load optimal window size for specific classifier
     */
    public static int loadOptimalWindowSize(Path baseDir, MLEngine.ClassifierType classifier) {
        Path configFile = baseDir.resolve("results/2_window_optimization/optimal_windows.txt");
        
        if (!Files.exists(configFile)) {
            return 1000; // Default to 1s if not found
        }
        
        try (BufferedReader reader = Files.newBufferedReader(configFile)) {
            String searchKey = classifier.name() + "_window_ms=";
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.startsWith(searchKey)) {
                    return Integer.parseInt(line.substring(searchKey.length()));
                }
            }
        } catch (Exception e) {
            // Ignore errors, return default
        }
        
        return 1000;
    }
    
    /**
     * Load all optimal windows
     */
    public static Map<MLEngine.ClassifierType, Integer> loadAllOptimalWindows(Path baseDir) {
        Map<MLEngine.ClassifierType, Integer> windows = new LinkedHashMap<>();
        for (MLEngine.ClassifierType classifier : MLEngine.ClassifierType.values()) {
            windows.put(classifier, loadOptimalWindowSize(baseDir, classifier));
        }
        return windows;
    }
    
    /**
     * Save optimal window sizes per classifier
     */
    private static void saveOptimalWindowSizes(Path resultsDir, 
                                               Map<MLEngine.ClassifierType, Integer> optimalWindows,
                                               Map<MLEngine.ClassifierType, Double> optimalAccuracies) throws IOException {
        Path configFile = resultsDir.resolve("optimal_windows.txt");
        
        try (BufferedWriter writer = Files.newBufferedWriter(configFile)) {
            writer.write("# Optimal window sizes per classifier");
            writer.newLine();
            writer.newLine();
            
            for (MLEngine.ClassifierType classifier : MLEngine.ClassifierType.values()) {
                int windowMs = optimalWindows.get(classifier);
                double accuracy = optimalAccuracies.get(classifier);
                
                writer.write(String.format("# %s%n", classifier.displayName));
                writer.write(String.format("%s_window_ms=%d%n", classifier.name(), windowMs));
                writer.write(String.format("%s_window_sec=%.1f%n", classifier.name(), windowMs/1000.0));
                writer.write(String.format("%s_accuracy=%.4f%n", classifier.name(), accuracy));
                writer.newLine();
            }
        }
        
        System.out.println("\nConfiguration saved to: " + configFile);
    }
    
    /**
     * Perform Sequential Feature Selection for a single classifier
     * Based on Part4_FeatureSelection logic
     */
    /**
     * Perform Sequential Feature Selection with custom output directory
     */
    public static SFSResult performSequentialFeatureSelection(Path csvFile, MLEngine.ClassifierType classifier, 
                                                              Path outputDir) throws Exception {
        System.out.printf("%n=== Running SFS for %s ===%n", classifier.displayName);
        
        // Read CSV data
        String[][] csvData = MyWekaUtils.readCSV(csvFile.toString());
        if (csvData == null || csvData.length == 0) {
            throw new IOException("CSV file is empty or unreadable");
        }
        
        // Perform SFS
        SFSResult result = performSFSCore(csvData, classifier);
        
        // Generate detailed evaluation report for final SFS result
        System.out.println("Generating detailed evaluation report for selected features...");
        Path sfsReportDir = outputDir.resolve(String.format("sfs_%s", classifier.name().toLowerCase()));
        Files.createDirectories(sfsReportDir);
        
        saveSFSResults(csvData, result.selectedFeatures, sfsReportDir, classifier);
        
        return result;
    }
    
    /**
     * Perform Sequential Feature Selection (uses csvFile parent directory for output)
     */
    public static SFSResult performSequentialFeatureSelection(Path csvFile, MLEngine.ClassifierType classifier) 
            throws Exception {
        return performSequentialFeatureSelection(csvFile, classifier, csvFile.getParent());
    }
    
    /**
     * Core SFS logic - select features iteratively
     */
    private static SFSResult performSFSCore(String[][] csvData, MLEngine.ClassifierType classifier) 
            throws Exception {
        
        // Get number of features (exclude class column)
        int numFeatures = csvData[0].length - 1;
        
        List<Integer> allFeatures = new ArrayList<>();
        for (int i = 0; i < numFeatures; i++) {
            allFeatures.add(i);
        }
        
        List<Integer> selectedFeatures = new ArrayList<>();
        double lastBestAccuracy = 0.0;
        
        System.out.println("Starting Sequential Forward Selection...");
        
        while (selectedFeatures.size() < allFeatures.size()) {
            int bestFeatureToAdd = -1;
            double bestAccuracyThisRound = -1.0;
            
            // Try each remaining feature
            for (int feature : allFeatures) {
                if (selectedFeatures.contains(feature)) continue;
                
                // Create trial subset
                List<Integer> trialSubset = new ArrayList<>(selectedFeatures);
                trialSubset.add(feature);
                
                // Evaluate this subset
                double accuracy = MLEngine.evaluateFeatureSubset(csvData, trialSubset, classifier);
                
                if (accuracy > bestAccuracyThisRound) {
                    bestAccuracyThisRound = accuracy;
                    bestFeatureToAdd = feature;
                }
            }
            
            // Check improvement threshold
            double improvement = bestAccuracyThisRound - lastBestAccuracy;
            
            if (bestFeatureToAdd == -1 || improvement < MIN_IMPROVEMENT) {
                System.out.printf("Stopping SFS: improvement (%.4f) below threshold (%.4f)%n", 
                                 improvement, MIN_IMPROVEMENT);
                break;
            }
            
            // Add the best feature
            selectedFeatures.add(bestFeatureToAdd);
            lastBestAccuracy = bestAccuracyThisRound;
            
            System.out.printf("Added feature %d -> %d features selected, accuracy: %.4f%n", 
                             bestFeatureToAdd, selectedFeatures.size(), lastBestAccuracy);
        }
        
        System.out.printf("%n✓ SFS completed: %d features selected, final accuracy: %.4f%n", 
                         selectedFeatures.size(), lastBestAccuracy);
        
        return new SFSResult(selectedFeatures, lastBestAccuracy);
    }
    
    /**
     * Save SFS results to files
     */
    private static void saveSFSResults(String[][] csvData, List<Integer> selectedFeatures,
                                      Path sfsReportDir, MLEngine.ClassifierType classifier) 
            throws Exception {
        
        // Save selected feature CSV (features first, class last - proper CSV format)
        Path sfsFeaturesCsv = sfsReportDir.resolve(String.format("sfs_%s_features.csv", classifier.name().toLowerCase()));
        try (PrintWriter writer = new PrintWriter(Files.newBufferedWriter(sfsFeaturesCsv))) {
            // Write header - features first, then class column
            boolean first = true;
            for (int featureIdx : selectedFeatures) {
                if (!first) writer.print(",");
                writer.print(csvData[0][featureIdx]);
                first = false;
            }
            writer.print("," + csvData[0][csvData[0].length - 1]); // class column last
            writer.println();
            
            // Write data rows - features first, then class value
            for (int row = 1; row < csvData.length; row++) {
                first = true;
                for (int featureIdx : selectedFeatures) {
                    if (!first) writer.print(",");
                    writer.print(csvData[row][featureIdx]);
                    first = false;
                }
                writer.print("," + csvData[row][csvData[row].length - 1]); // class value last
                writer.println();
            }
        }
        
        // Generate detailed report
        String experimentName = String.format("SFS_%s_%dfeatures", classifier.name(), selectedFeatures.size());
        MLEngine.evaluateClassifier(sfsFeaturesCsv, classifier, sfsReportDir, experimentName);
    }
    
    /**
     * Run SFS for all classifiers (J48, RandomForest, SVM)
     */
    public static Map<MLEngine.ClassifierType, SFSResult> runComprehensiveFeatureSelection(Path csvFile) 
            throws Exception {
        System.out.println("=== Sequential Feature Selection for All Classifiers ===");
        
        Map<MLEngine.ClassifierType, SFSResult> results = new LinkedHashMap<>();
        
        for (MLEngine.ClassifierType classifier : MLEngine.ClassifierType.values()) {
            System.out.println();
            System.out.println("--- " + classifier.displayName + " ---");
            
            try {
                SFSResult result = performSequentialFeatureSelection(csvFile, classifier);
                results.put(classifier, result);
            } catch (Exception e) {
                System.err.printf("Error running SFS for %s: %s%n", classifier, e.getMessage());
                results.put(classifier, new SFSResult(new ArrayList<>(), 0.0));
            }
        }
        
        // Print summary
        System.out.println();
        System.out.println("=== Feature Selection Summary ===");
        for (Map.Entry<MLEngine.ClassifierType, SFSResult> entry : results.entrySet()) {
            SFSResult result = entry.getValue();
            System.out.printf("%-20s: %d features, accuracy: %.4f%n", 
                             entry.getKey().displayName, 
                             result.selectedFeatures.size(), 
                             result.accuracy);
        }
        
        return results;
    }
    
    /**
     * Window optimization result container - stores optimal window per classifier
     */
    public static class WindowOptimizationResult {
        public final Map<MLEngine.ClassifierType, Integer> optimalWindows;
        public final Map<MLEngine.ClassifierType, Double> optimalAccuracies;
        
        public WindowOptimizationResult(Map<MLEngine.ClassifierType, Integer> optimalWindows,
                                       Map<MLEngine.ClassifierType, Double> optimalAccuracies) {
            this.optimalWindows = new LinkedHashMap<>(optimalWindows);
            this.optimalAccuracies = new LinkedHashMap<>(optimalAccuracies);
        }
        
        public int getOptimalWindow(MLEngine.ClassifierType classifier) {
            return optimalWindows.getOrDefault(classifier, 1000);
        }
        
        public double getOptimalAccuracy(MLEngine.ClassifierType classifier) {
            return optimalAccuracies.getOrDefault(classifier, 0.0);
        }
    }
    
    /**
     * Sequential Feature Selection result container
     */
    public static class SFSResult {
        public final List<Integer> selectedFeatures;
        public final double accuracy;
        
        public SFSResult(List<Integer> selectedFeatures, double accuracy) {
            this.selectedFeatures = new ArrayList<>(selectedFeatures);
            this.accuracy = accuracy;
        }
    }
}
