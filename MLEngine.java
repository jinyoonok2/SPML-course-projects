import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * MLEngine - Machine Learning operations and classifier management
 * 
 * Responsibilities:
 * - Train and evaluate classifiers (J48, RandomForest, SVM)
 * - Cross-validation and performance metrics
 * - Feature subset evaluation
 * - Model comparison and selection
 * 
 * Wraps and extends MyWekaUtils with higher-level ML operations
 */
public class MLEngine {
    
    public enum ClassifierType {
        J48(1, "J48 Decision Tree"),
        RANDOM_FOREST(2, "Random Forest"),
        SVM(3, "Support Vector Machine");
        
        public final int wekaId;
        public final String displayName;
        
        ClassifierType(int wekaId, String displayName) {
            this.wekaId = wekaId;
            this.displayName = displayName;
        }
        
        @Override
        public String toString() {
            return displayName;
        }
    }
    
    /**
     * Modular experiment configuration
     * Allows any combination of features, window, classifier, and feature selection
     */
    public static class ExperimentConfig {
        public final boolean useExpandedFeatures;  // true=12 features, false=6 features
        public final int windowMs;                 // Window size in milliseconds (1000-4000)
        public final ClassifierType classifier;    // J48, RandomForest, or SVM
        public final boolean useFeatureSelection;  // true=run SFS, false=use all features
        public final String description;           // Human-readable description
        
        public ExperimentConfig(boolean useExpandedFeatures, int windowMs, 
                               ClassifierType classifier, boolean useFeatureSelection) {
            this(useExpandedFeatures, windowMs, classifier, useFeatureSelection, null);
        }
        
        public ExperimentConfig(boolean useExpandedFeatures, int windowMs, 
                               ClassifierType classifier, boolean useFeatureSelection,
                               String description) {
            this.useExpandedFeatures = useExpandedFeatures;
            this.windowMs = windowMs;
            this.classifier = classifier;
            this.useFeatureSelection = useFeatureSelection;
            this.description = description != null ? description : generateDescription();
        }
        
        private String generateDescription() {
            return String.format("%s features, %dms window, %s%s",
                useExpandedFeatures ? "12" : "6",
                windowMs,
                classifier.displayName,
                useFeatureSelection ? " with SFS" : "");
        }
        
        @Override
        public String toString() {
            return description;
        }
        
        // Factory methods for common configurations
        
        public static ExperimentConfig baseline() {
            return new ExperimentConfig(false, 1000, ClassifierType.J48, false, "Baseline");
        }
        
        public static ExperimentConfig withWindow(int windowMs, ClassifierType classifier) {
            return new ExperimentConfig(false, windowMs, classifier, false,
                String.format("Window optimization (%dms, %s)", windowMs, classifier.displayName));
        }
        
        public static ExperimentConfig withExpandedFeatures(int windowMs, ClassifierType classifier) {
            return new ExperimentConfig(true, windowMs, classifier, false,
                String.format("Expanded features (%dms, %s)", windowMs, classifier.displayName));
        }
        
        public static ExperimentConfig withFeatureSelection(int windowMs, ClassifierType classifier) {
            return new ExperimentConfig(true, windowMs, classifier, true,
                String.format("SFS (%dms, %s)", windowMs, classifier.displayName));
        }
        
        public static ExperimentConfig custom(boolean expanded, int windowMs, 
                                             ClassifierType classifier, boolean sfs) {
            return new ExperimentConfig(expanded, windowMs, classifier, sfs);
        }
    }
    
    /**
     * Evaluate classifier performance on CSV file
     * Converts to ARFF and evaluates using MyWekaUtils
     */
    public static double evaluateClassifier(Path csvFile, ClassifierType classifierType) throws Exception {
        if (!Files.exists(csvFile)) {
            throw new FileNotFoundException("CSV file not found: " + csvFile);
        }
        
        System.out.printf("Evaluating %s on %s...%n", classifierType.displayName, csvFile.getFileName());
        
        // Read CSV data
        String[][] csvData = MyWekaUtils.readCSV(csvFile.toString());
        
        // Convert to ARFF format (all features)
        int[] allFeatures = new int[csvData[0].length - 1]; // All features except class
        for (int i = 0; i < allFeatures.length; i++) {
            allFeatures[i] = i;
        }
        String arffData = MyWekaUtils.csvToArff(csvData, allFeatures);
        
        // Evaluate
        double accuracy = MyWekaUtils.classify(arffData, classifierType.wekaId) / 100.0; // Convert percentage to decimal
        
        System.out.printf("✓ %s accuracy: %.4f%n", classifierType.displayName, accuracy);
        return accuracy;
    }
    
    /**
     * Evaluate specific feature subset using CSV data
     * Used for Sequential Feature Selection
     */
    public static double evaluateFeatureSubset(String[][] csvData, List<Integer> featureIndices, ClassifierType classifierType) 
            throws Exception {
        // Convert List<Integer> to int[]
        int[] featIndices = featureIndices.stream().mapToInt(i -> i).toArray();
        
        // Convert to ARFF and evaluate
        String arffData = MyWekaUtils.csvToArff(csvData, featIndices);
        return MyWekaUtils.classify(arffData, classifierType.wekaId) / 100.0; // Convert percentage to decimal
    }
    
    /**
     * MODULAR EXPERIMENT RUNNER
     * Run any experiment configuration - mix and match features, windows, classifiers, SFS
     */
    public static ExperimentResult runConfigurableExperiment(Path baseDir, ExperimentConfig config) 
            throws Exception {
        System.out.println("=== Configurable Experiment: " + config.description + " ===");
        
        // Ensure data is formatted
        DataManager.formatRawData(baseDir);
        
        // Create results directory
        Path resultsDir = baseDir.resolve("results/configurable");
        Files.createDirectories(resultsDir);
        
        // Step 1: Extract features based on config
        Path featuresCsv = resultsDir.resolve(
            String.format("%s_%dms_%s.csv",
                config.useExpandedFeatures ? "expanded" : "basic",
                config.windowMs,
                config.classifier.name().toLowerCase()));
        
        System.out.printf("Extracting %s with %dms window...%n",
            config.useExpandedFeatures ? "12 features" : "6 features",
            config.windowMs);
        
        if (config.useExpandedFeatures) {
            FeatureEngine.extractExpandedFeatures(baseDir, featuresCsv, config.windowMs, 1000);
        } else {
            FeatureEngine.extractBasicFeatures(baseDir, featuresCsv, config.windowMs, 1000);
        }
        
        double accuracy;
        List<Integer> selectedFeatures = null;
        
        // Step 2: Optionally run feature selection
        if (config.useFeatureSelection) {
            System.out.printf("Running SFS for %s...%n", config.classifier.displayName);
            OptimizationEngine.SFSResult sfsResult = 
                OptimizationEngine.performSequentialFeatureSelection(featuresCsv, config.classifier);
            accuracy = sfsResult.accuracy;
            selectedFeatures = sfsResult.selectedFeatures;
            System.out.printf("✓ SFS selected %d features (accuracy: %.4f)%n", 
                selectedFeatures.size(), accuracy);
        } else {
            // Step 3: Evaluate with all features
            System.out.printf("Evaluating %s...%n", config.classifier.displayName);
            accuracy = evaluateClassifier(featuresCsv, config.classifier);
        }
        
        System.out.printf("✓ Final accuracy: %.4f%n%n", accuracy);
        
        return new ExperimentResult(config, accuracy, selectedFeatures);
    }
    
    /**
     * Result container for configurable experiments
     */
    public static class ExperimentResult {
        public final ExperimentConfig config;
        public final double accuracy;
        public final List<Integer> selectedFeatures; // null if SFS not used
        
        public ExperimentResult(ExperimentConfig config, double accuracy, List<Integer> selectedFeatures) {
            this.config = config;
            this.accuracy = accuracy;
            this.selectedFeatures = selectedFeatures;
        }
        
        @Override
        public String toString() {
            String features = selectedFeatures != null 
                ? String.format(" (%d features selected)", selectedFeatures.size())
                : "";
            return String.format("%s: %.4f%s", config.description, accuracy, features);
        }
    }
    
    /**
     * Compare all classifiers on the same dataset
     * Returns performance results for each classifier
     */
    public static Map<ClassifierType, Double> compareAllClassifiers(Path featureFile) throws Exception {
        System.out.println("=== Multi-Classifier Comparison ===");
        
        Map<ClassifierType, Double> results = new LinkedHashMap<>();
        
        for (ClassifierType classifier : ClassifierType.values()) {
            try {
                double accuracy = evaluateClassifier(featureFile, classifier);
                results.put(classifier, accuracy);
            } catch (Exception e) {
                System.err.printf("Error evaluating %s: %s%n", classifier, e.getMessage());
                results.put(classifier, 0.0);
            }
        }
        
        // Print comparison summary
        System.out.println();
        System.out.println("=== Classifier Comparison Results ===");
        ClassifierType bestClassifier = null;
        double bestAccuracy = 0.0;
        
        for (Map.Entry<ClassifierType, Double> entry : results.entrySet()) {
            ClassifierType classifier = entry.getKey();
            double accuracy = entry.getValue();
            
            System.out.printf("%-20s: %.4f%n", classifier.displayName, accuracy);
            
            if (accuracy > bestAccuracy) {
                bestAccuracy = accuracy;
                bestClassifier = classifier;
            }
        }
        
        if (bestClassifier != null) {
            System.out.printf("%n✓ Best Performer: %s (%.4f accuracy)%n", bestClassifier.displayName, bestAccuracy);
        }
        
        return results;
    }
    
    /**
     * Run baseline evaluation for ALL classifiers
     * Each classifier gets its own baseline: 6 features, 1s window, no SFS
     */
    public static Map<ClassifierType, Double> runBaselineEvaluation(Path baseDir) throws Exception {
        System.out.println("=== Baseline Evaluation (All Classifiers) ===");
        System.out.println("Configuration: 6 basic features, 1s window, no feature selection");
        System.out.println();
        
        // Create results directory
        Path resultsDir = baseDir.resolve("results/baseline");
        Files.createDirectories(resultsDir);
        
        // Format data and extract basic features once
        DataManager.formatRawData(baseDir);
        Path featuresCsv = resultsDir.resolve("baseline_features.csv");
        FeatureEngine.extractBasicFeatures(baseDir, featuresCsv, 1000, 1000);
        
        // Evaluate each classifier
        Map<ClassifierType, Double> baselineResults = new LinkedHashMap<>();
        
        for (ClassifierType classifier : ClassifierType.values()) {
            System.out.printf("Evaluating %s...%n", classifier.displayName);
            double accuracy = evaluateClassifier(featuresCsv, classifier);
            baselineResults.put(classifier, accuracy);
        }
        
        // Print summary
        System.out.println();
        System.out.println("=== Baseline Results ===");
        for (Map.Entry<ClassifierType, Double> entry : baselineResults.entrySet()) {
            System.out.printf("  %-20s: %.4f%n", entry.getKey().displayName, entry.getValue());
        }
        
        // Save results
        saveEvaluationResults(resultsDir, "baseline_results.txt", 
                             "Baseline Evaluation Results (All Classifiers)", 
                             baselineResults);
        
        System.out.println("✓ Baseline evaluation complete for all classifiers");
        return baselineResults;
    }
    
    /**
     * Run comprehensive evaluation pipeline
     * Combines all parts: baseline -> optimization -> feature selection -> comparison
     */
    public static ExperimentResults runCompleteExperiment(Path baseDir) throws Exception {
        System.out.println("======================================");
        System.out.println("    COMPLETE EXPERIMENT PIPELINE");
        System.out.println("======================================");
        System.out.println();
        
        ExperimentResults results = new ExperimentResults();
        
        // Step 1: Baseline evaluation (6 features, 1s window, all classifiers)
        System.out.println("=== STEP 1: Baseline Evaluation ===");
        results.baselineAccuracies = runBaselineEvaluation(baseDir);
        System.out.println();
        
        // Step 2: Window optimization (per-classifier with basic 6 features)
        System.out.println("=== STEP 2: Window Size Optimization ===");
        OptimizationEngine.WindowOptimizationResult windowResult = OptimizationEngine.findOptimalWindowSize(baseDir);
        results.optimalWindows = windowResult.optimalWindows;
        results.windowOptimizationAccuracies = windowResult.optimalAccuracies;
        System.out.println();
        
        // Step 3: Create Master Datasets (per-classifier with 12 features and optimal windows)
        System.out.println("=== STEP 3: Master Dataset Creation (Per-Classifier) ===");
        Path resultsDir = baseDir.resolve("results/experiment");
        Files.createDirectories(resultsDir);
        
        results.masterDatasets = new LinkedHashMap<>();
        
        for (ClassifierType classifier : ClassifierType.values()) {
            int optimalWindow = results.optimalWindows.get(classifier);
            Path masterDataset = resultsDir.resolve(
                String.format("%s_master_dataset_12features.csv", classifier.name().toLowerCase()));
            
            System.out.printf("Creating master dataset for %s (%.0fs window, 12 features)...%n",
                             classifier.displayName, optimalWindow/1000.0);
            
            FeatureEngine.extractExpandedFeatures(baseDir, masterDataset, optimalWindow, 1000);
            results.masterDatasets.put(classifier, masterDataset);
            
            System.out.printf("  ✓ %s%n", masterDataset.getFileName());
        }
        System.out.println();
        
        // Step 4: Sequential Feature Selection (per-classifier on their own master datasets)
        System.out.println("=== STEP 4: Sequential Feature Selection & Classifier Comparison ===");
        System.out.println("Running SFS on each classifier's master dataset...");
        System.out.println();
        
        results.sfsResults = new LinkedHashMap<>();
        
        for (ClassifierType classifier : ClassifierType.values()) {
            Path masterDataset = results.masterDatasets.get(classifier);
            System.out.printf("Running SFS for %s...%n", classifier.displayName);
            
            OptimizationEngine.SFSResult sfsResult = 
                OptimizationEngine.performSequentialFeatureSelection(masterDataset, classifier);
            results.sfsResults.put(classifier, sfsResult);
        }
        
        // Extract final accuracies from SFS results for comparison
        results.finalComparison = new LinkedHashMap<>();
        for (Map.Entry<ClassifierType, OptimizationEngine.SFSResult> entry : results.sfsResults.entrySet()) {
            results.finalComparison.put(entry.getKey(), entry.getValue().accuracy);
        }
        
        // Generate comprehensive report
        generateExperimentReport(baseDir, results);
        
        System.out.println();
        System.out.println("======================================");
        System.out.println("     EXPERIMENT COMPLETED!");
        System.out.println("======================================");
        
        return results;
    }
    
    /**
     * Save evaluation results to file
     */
    private static void saveEvaluationResults(Path resultsDir, String filename, String title, 
                                            Map<ClassifierType, Double> results) throws IOException {
        StringBuilder report = new StringBuilder();
        report.append("=== ").append(title).append(" ===").append(System.lineSeparator()).append(System.lineSeparator());
        report.append("Timestamp: ").append(new Date()).append(System.lineSeparator()).append(System.lineSeparator());
        
        for (Map.Entry<ClassifierType, Double> entry : results.entrySet()) {
            report.append(String.format("%-20s: %.4f%n", entry.getKey().displayName, entry.getValue()));
        }
        
        Files.write(resultsDir.resolve(filename), report.toString().getBytes());
        System.out.println("Results saved to: " + resultsDir.resolve(filename));
    }
    
    /**
     * Generate comprehensive experiment report
     */
    private static void generateExperimentReport(Path baseDir, ExperimentResults results) throws IOException {
        Path reportDir = baseDir.resolve("results/final_report");
        Files.createDirectories(reportDir);
        
        StringBuilder report = new StringBuilder();
        report.append("============================================").append(System.lineSeparator());
        report.append("    GESTURE RECOGNITION EXPERIMENT REPORT").append(System.lineSeparator());
        report.append("============================================").append(System.lineSeparator()).append(System.lineSeparator());
        report.append("Generated: ").append(new Date()).append(System.lineSeparator()).append(System.lineSeparator());
        
        report.append("1. BASELINE RESULTS").append(System.lineSeparator());
        report.append("-------------------").append(System.lineSeparator());
        for (Map.Entry<ClassifierType, Double> entry : results.baselineAccuracies.entrySet()) {
            report.append(String.format("%s: %.4f%n", entry.getKey().displayName, entry.getValue()));
        }
        report.append(System.lineSeparator());
        
        report.append("2. WINDOW OPTIMIZATION").append(System.lineSeparator());
        report.append("-----------------------").append(System.lineSeparator());
        for (Map.Entry<ClassifierType, Integer> entry : results.optimalWindows.entrySet()) {
            Double accuracy = results.windowOptimizationAccuracies.get(entry.getKey());
            report.append(String.format("%s: %ds window, accuracy: %.4f%n", 
                         entry.getKey().displayName, entry.getValue()/1000, accuracy));
        }
        report.append(System.lineSeparator());
        
        report.append("3. FEATURE EXPANSION").append(System.lineSeparator());
        report.append("--------------------").append(System.lineSeparator());
        for (Map.Entry<ClassifierType, Path> entry : results.masterDatasets.entrySet()) {
            report.append(String.format("%s: %s%n", entry.getKey().displayName, entry.getValue().getFileName()));
        }
        report.append(System.lineSeparator());
        
        report.append("4. FEATURE SELECTION RESULTS").append(System.lineSeparator());
        report.append("-----------------------------").append(System.lineSeparator());
        for (Map.Entry<ClassifierType, OptimizationEngine.SFSResult> entry : results.sfsResults.entrySet()) {
            OptimizationEngine.SFSResult sfs = entry.getValue();
            report.append(String.format("%s: %d features selected, accuracy: %.4f%n", 
                         entry.getKey().displayName, sfs.selectedFeatures.size(), sfs.accuracy));
        }
        report.append(System.lineSeparator());
        
        report.append("5. FINAL CLASSIFIER COMPARISON").append(System.lineSeparator());
        report.append("-------------------------------").append(System.lineSeparator());
        ClassifierType bestClassifier = null;
        double bestAccuracy = 0.0;
        
        for (Map.Entry<ClassifierType, Double> entry : results.finalComparison.entrySet()) {
            report.append(String.format("%s: %.4f%n", entry.getKey().displayName, entry.getValue()));
            if (entry.getValue() > bestAccuracy) {
                bestAccuracy = entry.getValue();
                bestClassifier = entry.getKey();
            }
        }
        
        if (bestClassifier != null) {
            report.append(String.format("%n✓ BEST PERFORMER: %s (%.4f accuracy)%n", bestClassifier.displayName, bestAccuracy));
        }
        
        report.append(System.lineSeparator()).append("============================================").append(System.lineSeparator());
        
        Files.write(reportDir.resolve("experiment_report.txt"), report.toString().getBytes());
        System.out.println();
        System.out.println("✓ Comprehensive report saved to: " + reportDir.resolve("experiment_report.txt"));
    }
    
    /**
     * Experiment results container
     */
    public static class ExperimentResults {
        public Map<ClassifierType, Double> baselineAccuracies;  // Baseline for each classifier
        public Map<ClassifierType, Integer> optimalWindows;      // Optimal window per classifier
        public Map<ClassifierType, Double> windowOptimizationAccuracies;  // Window opt accuracy per classifier
        public Map<ClassifierType, Path> masterDatasets;         // Master dataset path per classifier
        public Map<ClassifierType, OptimizationEngine.SFSResult> sfsResults;
        public Map<ClassifierType, Double> finalComparison;
    }
}