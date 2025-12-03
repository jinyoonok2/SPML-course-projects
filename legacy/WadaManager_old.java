import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;

/**
 * WadaManager - Updated to use new engine-based architecture
 * 
 * Maintains full backward compatibility with original command interface
 * while internally using the new well-distributed engine architecture:
 * - DataManager: Raw data processing and format conversion
 * - FeatureEngine: Feature extraction and engineering  
 * - OptimizationEngine: Parameter optimization and feature selection
 * - MLEngine: Machine learning operations and evaluation
 */
public class WadaManager {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		if (args.length < 1) {
			System.err.println("Usage: java WadaManager <command>");
			System.err.println("Commands: part1, part2, part3, part4, part5, all");
			System.err.println("Note: Use ExperimentRunner for new streamlined commands");
			System.exit(1);
		}
		
		String funcName = args[0];
		Path baseDir = Paths.get(System.getProperty("user.dir"));
		
		try {
			// Baseline experiments (Part1) - Now uses engine architecture
			if (funcName.equals("part1")) {
				System.out.println("=== PART 1: Baseline Experiments ===");
				double accuracy = MLEngine.runBaselineEvaluation(baseDir);
				System.out.printf("✓ Part 1 completed successfully (accuracy: %.4f)%n", accuracy);
			}
			
            // Window tuning (Part2) - Per-classifier window optimization
            else if (funcName.equals("part2")) {
                System.out.println("=== PART 2: Window Size Optimization ===");
                
                // Ensure data is formatted first
                DataManager.formatRawData(baseDir);
                
                OptimizationEngine.WindowOptimizationResult result = OptimizationEngine.findOptimalWindowSize(baseDir);
                System.out.printf("✓ Part 2 completed successfully%n");
                System.out.println("Per-classifier optimal windows:");
                for (Map.Entry<MLEngine.ClassifierType, Integer> entry : result.optimalWindows.entrySet()) {
                    double accuracy = result.optimalAccuracies.get(entry.getKey());
                    System.out.printf("  %-20s: %.0fs (accuracy: %.4f)%n", 
                                     entry.getKey().displayName, entry.getValue()/1000.0, accuracy);
                }
            }            // Feature expansion (Part3) - Uses J48 optimal window
            else if (funcName.equals("part3")) {
                System.out.println("=== PART 3: Feature Expansion ===");
                
                // Ensure data is formatted
                DataManager.formatRawData(baseDir);
                
                // Get optimal window size for J48 (run optimization if needed)
                int optimalWindow = OptimizationEngine.loadOptimalWindowSize(baseDir, MLEngine.ClassifierType.J48);
                if (optimalWindow == 1000) {
                    System.out.println("No window optimization found, running optimization first...");
                    OptimizationEngine.WindowOptimizationResult result = OptimizationEngine.findOptimalWindowSize(baseDir);
                    optimalWindow = result.getOptimalWindow(MLEngine.ClassifierType.J48);
                }
                
                // Extract expanded features
                Path resultsDir = baseDir.resolve("results/feature_expansion");
                Files.createDirectories(resultsDir);
                Path expandedCsv = resultsDir.resolve("j48_expanded_features.csv");
                
                FeatureEngine.extractExpandedFeatures(baseDir, expandedCsv, optimalWindow, 1000);
                
                // Evaluate expanded features
                double accuracy = MLEngine.evaluateClassifier(expandedCsv, MLEngine.ClassifierType.J48);
                
                System.out.printf("✓ Part 3 completed successfully%n");
                System.out.printf("  12 features extracted with %.0fs window (J48 optimal, accuracy: %.4f)%n", 
                                 optimalWindow/1000.0, accuracy);
            }            // Feature selection (Part4) - SFS with J48 using its optimal window
            else if (funcName.equals("part4")) {
                System.out.println("=== PART 4: Feature Selection (J48) ===");
                
                DataManager.formatRawData(baseDir);
                
                // Get J48 optimal window
                int optimalWindow = OptimizationEngine.loadOptimalWindowSize(baseDir, MLEngine.ClassifierType.J48);
                if (optimalWindow == 1000) {
                    System.out.println("No window optimization found, running optimization first...");
                    OptimizationEngine.WindowOptimizationResult windowResult = OptimizationEngine.findOptimalWindowSize(baseDir);
                    optimalWindow = windowResult.getOptimalWindow(MLEngine.ClassifierType.J48);
                }
                
                // Extract expanded features with J48 optimal window
                Path resultsDir = baseDir.resolve("results/feature_selection");
                Files.createDirectories(resultsDir);
                Path expandedCsv = resultsDir.resolve("j48_expanded_features.csv");
                
                System.out.printf("Extracting 12 features with %.0fs window...%n", optimalWindow/1000.0);
                FeatureEngine.extractExpandedFeatures(baseDir, expandedCsv, optimalWindow, 1000);
                
                // Run SFS for J48
                OptimizationEngine.SFSResult result = 
                    OptimizationEngine.performSequentialFeatureSelection(expandedCsv, MLEngine.ClassifierType.J48);
                
                System.out.printf("✓ Part 4 completed successfully%n");
                System.out.printf("  J48: %d features selected (accuracy: %.4f)%n", 
                                result.selectedFeatures.size(), result.accuracy);
            }            // Classifier comparison (Part5) - SFS for all classifiers with per-classifier windows
            else if (funcName.equals("part5")) {
                System.out.println("=== PART 5: Classifier Comparison (SFS) ===");
                
                DataManager.formatRawData(baseDir);
                
                // Load optimal windows or run optimization
                Map<MLEngine.ClassifierType, Integer> optimalWindows = OptimizationEngine.loadAllOptimalWindows(baseDir);
                boolean needsOptimization = optimalWindows.values().stream().allMatch(w -> w == 1000);
                
                if (needsOptimization) {
                    System.out.println("No window optimization found, running optimization first...");
                    OptimizationEngine.WindowOptimizationResult windowResult = OptimizationEngine.findOptimalWindowSize(baseDir);
                    optimalWindows = windowResult.optimalWindows;
                }
                
                // Run SFS for each classifier with its optimal window
                Path resultsDir = baseDir.resolve("results/part5");
                Files.createDirectories(resultsDir);
                
                Map<MLEngine.ClassifierType, OptimizationEngine.SFSResult> sfsResults = new LinkedHashMap<>();
                
                for (MLEngine.ClassifierType classifier : MLEngine.ClassifierType.values()) {
                    int optimalWindow = optimalWindows.get(classifier);
                    Path expandedCsv = resultsDir.resolve(String.format("%s_expanded_features.csv", 
                                                                        classifier.name().toLowerCase()));
                    
                    System.out.printf("%nProcessing %s (%.0fs window)...%n", 
                                     classifier.displayName, optimalWindow/1000.0);
                    
                    FeatureEngine.extractExpandedFeatures(baseDir, expandedCsv, optimalWindow, 1000);
                    OptimizationEngine.SFSResult result = 
                        OptimizationEngine.performSequentialFeatureSelection(expandedCsv, classifier);
                    sfsResults.put(classifier, result);
                }
                
                System.out.printf("✓ Part 5 completed successfully%n");
                System.out.println("\nFinal Results:");
                
                // Print results and find best
                MLEngine.ClassifierType bestClassifier = null;
                double bestAccuracy = 0.0;
                
                for (Map.Entry<MLEngine.ClassifierType, OptimizationEngine.SFSResult> entry : sfsResults.entrySet()) {
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
            }			// Run all parts - Now uses complete experiment pipeline
			else if (funcName.equals("all")) {
				System.out.println("=== RUNNING ALL PARTS ===");
				
				MLEngine.ExperimentResults results = MLEngine.runCompleteExperiment(baseDir);
				
				System.out.println("\n=== EXPERIMENT SUMMARY ===");
				System.out.printf("Part 1 - Baseline accuracy: %.4f%n", results.baselineAccuracy);
				System.out.printf("Part 2 - Optimal window: %.0fs (accuracy: %.4f)%n", 
								 results.optimalWindowMs/1000.0, results.windowOptimizationAccuracy);
				System.out.printf("Part 3 - Expanded features accuracy: %.4f%n", results.expandedFeaturesAccuracy);
				System.out.println("Part 4 - Feature selection completed for all classifiers");
				System.out.println("Part 5 - Classifier comparison completed");
				
				// Find best overall result
				double bestAccuracy = 0.0;
				String bestConfig = "";
				
				for (Map.Entry<MLEngine.ClassifierType, Double> entry : results.finalComparison.entrySet()) {
					if (entry.getValue() > bestAccuracy) {
						bestAccuracy = entry.getValue();
						bestConfig = entry.getKey().toString();
					}
				}
				
				System.out.printf("%n🏆 BEST RESULT: %s with %.4f accuracy%n", bestConfig, bestAccuracy);
				System.out.println("✓ All parts completed successfully using new engine architecture");
			}
			
			else {
				System.err.println("Unknown command: " + funcName);
				System.err.println("Available commands: part1, part2, part3, part4, part5, all");
				System.err.println("For new streamlined commands, use: java ExperimentRunner help");
				System.exit(1);
			}
			
		} catch (Exception e) {
			System.err.println("Error executing " + funcName + ": " + e.getMessage());
			e.printStackTrace();
			System.exit(1);
		}
	}
}