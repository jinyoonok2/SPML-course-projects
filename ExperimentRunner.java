import java.nio.file.*;
import java.util.*;

/**
 * ExperimentRunner - Main orchestration and command interface
 * 
 * This is the new main entry point replacing the old WadaManager.
 * Supports modern streamlined commands for running ML experiments.
 * 
 * Responsibilities:
 * - Command-line interface and argument parsing
 * - Pipeline orchestration and workflow management
 * - Integration between all engine components
 * 
 * Architecture:
 * - DataManager: Raw data processing and format conversion
 * - FeatureEngine: Feature extraction and engineering
 * - OptimizationEngine: Parameter optimization and feature selection
 * - MLEngine: Machine learning operations and evaluation
 * - ExperimentRunner: Orchestration and command interface (this class)
 */
public class ExperimentRunner {

    public static void main(String[] args) {
        if (args.length < 1) {
            showHelp();
            System.exit(1);
        }
        
        String command = args[0];
        Path baseDir = Paths.get(".").toAbsolutePath().normalize();
        
        try {
            switch (command.toLowerCase()) {
                // New streamlined commands
                case "baseline":
                    runBaseline(baseDir);
                    break;
                case "optimize":
                    runOptimization(baseDir);
                    break;
                case "features":
                    runFeatureExpansion(baseDir);
                    break;
                case "selection":
                    runFeatureSelection(baseDir);
                    break;
                case "compare":
                    runFeatureSelection(baseDir); // SFS IS the comparison
                    break;
                case "experiment":
                    runCompleteExperiment(baseDir);
                    break;
                    
                // Individual operations
                case "format":
                    DataManager.formatRawData(baseDir);
                    break;
                case "extract":
                    runBasicFeatureExtraction(baseDir);
                    break;
                    
                // NEW: Custom configurable experiment
                case "custom":
                    runCustomExperiment(baseDir, args);
                    break;
                    
                case "help":
                    showHelp();
                    break;
                    
                default:
                    System.err.println("Unknown command: " + command);
                    showHelp();
                    System.exit(1);
            }
        } catch (Exception e) {
            System.err.println("Error executing command '" + command + "': " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    /**
     * Run baseline evaluation (Part 1 equivalent)
     */
    private static void runBaseline(Path baseDir) throws Exception {
        Map<MLEngine.ClassifierType, Double> results = MLEngine.runBaselineEvaluation(baseDir);
        
        System.out.printf("%n✓ Baseline evaluation completed successfully%n");
        System.out.println("Each classifier tested with 6 features, 1s window, no feature selection");
    }
    
    /**
     * Run window size optimization (Part 2 equivalent)
     */
    private static void runOptimization(Path baseDir) throws Exception {
        System.out.println("=== WINDOW SIZE OPTIMIZATION ===");
        
        DataManager.formatRawData(baseDir);
        
        OptimizationEngine.WindowOptimizationResult result = OptimizationEngine.findOptimalWindowSize(baseDir);
        
        System.out.printf("%n✓ Window optimization completed successfully%n");
        System.out.println("Optimal windows per classifier:");
        for (Map.Entry<MLEngine.ClassifierType, Integer> entry : result.optimalWindows.entrySet()) {
            double accuracy = result.optimalAccuracies.get(entry.getKey());
            System.out.printf("  %-20s: %.0fs (accuracy: %.4f)%n",
                             entry.getKey().displayName, entry.getValue()/1000.0, accuracy);
        }
    }
    
    /**
     * Run feature expansion with optimal windows (Part 3 equivalent)
     * Creates master dataset per classifier with its optimal window
     */
    private static void runFeatureExpansion(Path baseDir) throws Exception {
        System.out.println("=== MASTER DATASET CREATION (PER-CLASSIFIER) ===");
        
        DataManager.formatRawData(baseDir);
        
        // Load optimal windows
        Map<MLEngine.ClassifierType, Integer> optimalWindows = OptimizationEngine.loadAllOptimalWindows(baseDir);
        boolean needsOptimization = optimalWindows.values().stream().allMatch(w -> w == 1000);
        
        if (needsOptimization) {
            System.out.println("No window optimization found, running optimization first...");
            OptimizationEngine.WindowOptimizationResult result = OptimizationEngine.findOptimalWindowSize(baseDir);
            optimalWindows = result.optimalWindows;
        }
        
        Path resultsDir = baseDir.resolve("results/feature_expansion");
        Files.createDirectories(resultsDir);
        
        for (MLEngine.ClassifierType classifier : MLEngine.ClassifierType.values()) {
            int optimalWindow = optimalWindows.get(classifier);
            Path masterDataset = resultsDir.resolve(
                String.format("%s_master_dataset_12features.csv", classifier.name().toLowerCase()));
            
            System.out.printf("Creating master dataset for %s (%.0fs window, 12 features)...%n",
                             classifier.displayName, optimalWindow/1000.0);
            
            FeatureEngine.extractExpandedFeatures(baseDir, masterDataset, optimalWindow, 1000);
            System.out.printf("  ✓ %s%n", masterDataset.getFileName());
        }
        
        System.out.printf("%n✓ Master datasets created successfully%n");
        System.out.println("Each classifier has its own 12-feature dataset with its optimal window");
    }
    
    /**
     * Run feature selection on per-classifier master datasets (Part 4 & 5 equivalent)
     * Runs SFS on each classifier's optimized master dataset
     */
    private static void runFeatureSelection(Path baseDir) throws Exception {
        System.out.println("=== FEATURE SELECTION & CLASSIFIER COMPARISON ===");
        
        // Check if master datasets exist
        Path resultsDir = baseDir.resolve("results/feature_expansion");
        boolean hasMasterDatasets = true;
        
        for (MLEngine.ClassifierType classifier : MLEngine.ClassifierType.values()) {
            Path masterDataset = resultsDir.resolve(
                String.format("%s_master_dataset_12features.csv", classifier.name().toLowerCase()));
            if (!Files.exists(masterDataset)) {
                hasMasterDatasets = false;
                break;
            }
        }
        
        if (!hasMasterDatasets) {
            System.out.println("Master datasets not found, creating them first...");
            runFeatureExpansion(baseDir);
        }
        
        // Run SFS for each classifier on its own master dataset
        System.out.println("Running SFS on each classifier's master dataset...");
        System.out.println();
        
        Map<MLEngine.ClassifierType, OptimizationEngine.SFSResult> results = new LinkedHashMap<>();
        
        for (MLEngine.ClassifierType classifier : MLEngine.ClassifierType.values()) {
            Path masterDataset = resultsDir.resolve(
                String.format("%s_master_dataset_12features.csv", classifier.name().toLowerCase()));
            
            System.out.printf("Running SFS for %s...%n", classifier.displayName);
            OptimizationEngine.SFSResult result = 
                OptimizationEngine.performSequentialFeatureSelection(masterDataset, classifier);
            results.put(classifier, result);
        }
        
        // Print final comparison
        System.out.printf("%n✓ Feature selection completed successfully%n");
        System.out.println("\nFinal Classifier Comparison:");
        
        MLEngine.ClassifierType bestClassifier = null;
        double bestAccuracy = 0.0;
        
        for (Map.Entry<MLEngine.ClassifierType, OptimizationEngine.SFSResult> entry : results.entrySet()) {
            OptimizationEngine.SFSResult result = entry.getValue();
            System.out.printf("  %-20s: %d features, accuracy: %.4f%n", 
                            entry.getKey().displayName, result.selectedFeatures.size(), result.accuracy);
            
            if (result.accuracy > bestAccuracy) {
                bestAccuracy = result.accuracy;
                bestClassifier = entry.getKey();
            }
        }
        
        if (bestClassifier != null) {
            System.out.printf("%n>>> Best classifier: %s (accuracy: %.4f)%n", 
                             bestClassifier.displayName, bestAccuracy);
        }
    }
    
    /**
     * Run complete experiment pipeline
     */
    private static void runCompleteExperiment(Path baseDir) throws Exception {
        MLEngine.ExperimentResults results = MLEngine.runCompleteExperiment(baseDir);
        
        System.out.println();
        System.out.println("=== EXPERIMENT SUMMARY ===");
        System.out.println("Step 1 - Baseline (6 features, 1s window, all classifiers):");
        for (Map.Entry<MLEngine.ClassifierType, Double> entry : results.baselineAccuracies.entrySet()) {
            System.out.printf("  %-20s: %.4f%n", entry.getKey().displayName, entry.getValue());
        }
        
        System.out.println("\nStep 2 - Optimal Windows (per classifier, 6 features):");
        for (Map.Entry<MLEngine.ClassifierType, Integer> entry : results.optimalWindows.entrySet()) {
            double accuracy = results.windowOptimizationAccuracies.get(entry.getKey());
            System.out.printf("  %-20s: %.0fs (accuracy: %.4f)%n",
                             entry.getKey().displayName, entry.getValue()/1000.0, accuracy);
        }
        
        System.out.println("\nStep 3 - Master Datasets (per classifier, 12 features, optimal windows):");
        for (Map.Entry<MLEngine.ClassifierType, Path> entry : results.masterDatasets.entrySet()) {
            System.out.printf("  %-20s: %s%n", entry.getKey().displayName, entry.getValue().getFileName());
        }
        
        System.out.println("\nStep 4 - SFS Results (per-classifier master datasets):");
        
        double bestAccuracy = 0.0;
        String bestConfig = "";
        
        for (Map.Entry<MLEngine.ClassifierType, Double> entry : results.finalComparison.entrySet()) {
            if (entry.getValue() > bestAccuracy) {
                bestAccuracy = entry.getValue();
                bestConfig = entry.getKey().toString();
            }
        }
        
        System.out.printf("%n🏆 BEST RESULT: %s with %.4f accuracy%n", bestConfig, bestAccuracy);
    }
    
    /**
     * Extract basic features only
     */
    private static void runBasicFeatureExtraction(Path baseDir) throws Exception {
        DataManager.formatRawData(baseDir);
        
        Path resultsDir = baseDir.resolve("results/basic_features");
        Files.createDirectories(resultsDir);
        Path featuresCsv = resultsDir.resolve("basic_features.csv");
        
        FeatureEngine.extractBasicFeatures(baseDir, featuresCsv, 1000, 1000);
        
        System.out.println("✓ Basic feature extraction completed");
        System.out.println("  Features saved to: " + featuresCsv);
    }
    
    /**
     * Run custom configurable experiment
     * Usage: java ExperimentRunner custom <features> <window> <classifier> [sfs]
     * Examples:
     *   java ExperimentRunner custom basic 2000 J48
     *   java ExperimentRunner custom expanded 3000 SVM sfs
     */
    private static void runCustomExperiment(Path baseDir, String[] args) throws Exception {
        if (args.length < 4) {
            System.err.println("Usage: java ExperimentRunner custom <features> <window> <classifier> [sfs]");
            System.err.println();
            System.err.println("Parameters:");
            System.err.println("  <features>    - 'basic' (6 features) or 'expanded' (12 features)");
            System.err.println("  <window>      - Window size in milliseconds (1000-4000)");
            System.err.println("  <classifier>  - 'J48', 'RF', or 'SVM'");
            System.err.println("  [sfs]         - Optional: add 'sfs' to enable feature selection");
            System.err.println();
            System.err.println("Examples:");
            System.err.println("  java ExperimentRunner custom basic 2000 J48");
            System.err.println("  java ExperimentRunner custom expanded 3000 SVM sfs");
            System.exit(1);
        }
        
        // Parse arguments
        boolean useExpanded = args[1].equalsIgnoreCase("expanded");
        int windowMs = Integer.parseInt(args[2]);
        
        MLEngine.ClassifierType classifier;
        String classifierArg = args[3].toUpperCase();
        if (classifierArg.equals("RF")) {
            classifier = MLEngine.ClassifierType.RANDOM_FOREST;
        } else if (classifierArg.equals("SVM")) {
            classifier = MLEngine.ClassifierType.SVM;
        } else {
            classifier = MLEngine.ClassifierType.J48;
        }
        
        boolean useSfs = args.length > 4 && args[4].equalsIgnoreCase("sfs");
        
        // Create configuration
        MLEngine.ExperimentConfig config = MLEngine.ExperimentConfig.custom(
            useExpanded, windowMs, classifier, useSfs);
        
        // Run experiment
        MLEngine.ExperimentResult result = MLEngine.runConfigurableExperiment(baseDir, config);
        
        System.out.println();
        System.out.println("=== CUSTOM EXPERIMENT COMPLETED ===");
        System.out.println("Configuration: " + result.config);
        System.out.printf("Final Accuracy: %.4f%n", result.accuracy);
        if (result.selectedFeatures != null) {
            System.out.println("Selected Features: " + result.selectedFeatures);
        }
    }
    
    /**
     * Show help message
     */
    private static void showHelp() {
        System.out.println("ExperimentRunner - Gesture Recognition Experiment Suite");
        System.out.println("======================================================");
        System.out.println();
        System.out.println("Usage: java ExperimentRunner <command>");
        System.out.println();
        System.out.println("=== COMMANDS ===");
        System.out.println("  baseline      - Complete baseline evaluation (6 features, 1s, J48)");
        System.out.println("  optimize      - Window size optimization (find best window per classifier)");
        System.out.println("  features      - Feature expansion with optimal window (12 features)");
        System.out.println("  selection     - Sequential feature selection for all classifiers");
        System.out.println("  compare       - Same as 'selection' (SFS comparison for all classifiers)");
        System.out.println("  experiment    - Run complete experiment pipeline (all steps)");
        System.out.println();
        System.out.println("=== INDIVIDUAL OPERATIONS ===");
        System.out.println("  format        - Format raw CSV data files only");
        System.out.println("  extract       - Extract basic features only (6 features, 1s window)");
        System.out.println();
        System.out.println("=== MODULAR EXPERIMENT (NEW!) ===");
        System.out.println("  custom <features> <window> <classifier> [sfs]");
        System.out.println("    Mix and match any configuration you want!");
        System.out.println("    <features>    - 'basic' (6) or 'expanded' (12)");
        System.out.println("    <window>      - Window size in ms (1000-4000)");
        System.out.println("    <classifier>  - 'J48', 'RF', or 'SVM'");
        System.out.println("    [sfs]         - Optional: add 'sfs' for feature selection");
        System.out.println();
        System.out.println("  help          - Show this help message");
        System.out.println();
        System.out.println("=== ARCHITECTURE ===");
        System.out.println("  DataManager        - Raw data processing and format conversion");
        System.out.println("  FeatureEngine      - Feature extraction and engineering");
        System.out.println("  OptimizationEngine - Parameter optimization and feature selection");
        System.out.println("  MLEngine           - Machine learning operations and evaluation");
        System.out.println("  MyWekaUtils        - Core Weka utilities (unchanged)");
        System.out.println();
        System.out.println("Examples:");
        System.out.println("  java ExperimentRunner experiment              # Run complete pipeline");
        System.out.println("  java ExperimentRunner baseline                # Quick baseline");
        System.out.println("  java ExperimentRunner custom basic 2000 J48   # 6 features, 2s window, J48");
        System.out.println("  java ExperimentRunner custom expanded 3000 SVM sfs  # 12 features, 3s, SVM with SFS");
        System.out.println("  java ExperimentRunner baseline      # Quick baseline evaluation");
        System.out.println("  java ExperimentRunner optimize      # Find optimal window size");
    }
}
