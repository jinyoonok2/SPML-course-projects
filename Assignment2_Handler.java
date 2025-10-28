import java.io.*;

/**
 * Assignment2_Handler
 * -------------------
 * Main entry point for Assignment 2 - Gesture Recognition Optimization
 * 
 * This handler coordinates all 5 parts of the assignment:
 *   Part 1: Baseline accuracy with all data (Decision Tree, 1s window, 6 features)
 *   Part 2: Time window tuning (1s, 2s, 3s, 4s)
 *   Part 3: Feature expansion (add median & RMS -> 12 features)
 *   Part 4: Sequential feature selection (Decision Tree)
 *   Part 5: Classifier comparison (Random Forest & SVM)
 * 
 * Usage:
 *   java -cp ".:lib/weka.jar" Assignment2_Handler <command>
 * 
 * Commands:
 *   part1  - Run Part 1 only
 *   part2  - Run Part 2 only
 *   part3  - Run Part 3 only
 *   part4  - Run Part 4 only
 *   part5  - Run Part 5 only
 *   all    - Run all parts in sequence
 */
public class Assignment2_Handler {

    public static void main(String[] args) {
        if (args.length < 1) {
            printUsage();
            System.exit(1);
        }

        String command = args[0].toLowerCase();
        
        try {
            switch (command) {
                case "part1":
                    runPart1();
                    break;
                case "part2":
                    runPart2();
                    break;
                case "part3":
                    runPart3();
                    break;
                case "part4":
                    runPart4();
                    break;
                case "part5":
                    runPart5();
                    break;
                case "all":
                    runAll();
                    break;
                default:
                    System.err.println("Unknown command: " + command);
                    printUsage();
                    System.exit(1);
            }
        } catch (Exception e) {
            System.err.println("Error executing " + command + ": " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    private static void runPart1() throws Exception {
        printHeader("PART 1: Baseline Accuracy with Complete Dataset");
        System.out.println("Processing all raw data with Decision Tree classifier");
        System.out.println("Window: 1 second");
        System.out.println("Features: 6 (mean, std per axis)");
        System.out.println();
        
        Part1_DataProcessor.run();
        
        System.out.println();
        System.out.println("Part 1 completed successfully!");
        System.out.println("=" .repeat(80));
    }

    private static void runPart2() throws Exception {
        printHeader("PART 2: Time Window Tuning");
        System.out.println("Testing window sizes: 1s, 2s, 3s, 4s (with 1s sliding window)");
        System.out.println("Features: 6 (mean, std per axis)");
        System.out.println("Classifier: Decision Tree");
        System.out.println();
        
        Part2_WindowTuning.run();
        
        System.out.println();
        System.out.println("Part 2 completed successfully!");
        System.out.println("=" .repeat(80));
    }

    private static void runPart3() throws Exception {
        printHeader("PART 3: Feature Expansion");
        System.out.println("Adding median and RMS features");
        System.out.println("Features: 12 (mean, std, median, RMS per axis)");
        System.out.println("Classifier: Decision Tree");
        System.out.println();
        
        Part3_FeatureExpansion.run();
        
        System.out.println();
        System.out.println("Part 3 completed successfully!");
        System.out.println("=" .repeat(80));
    }

    private static void runPart4() throws Exception {
        printHeader("PART 4: Sequential Feature Selection (Decision Tree)");
        System.out.println("Finding optimal feature subset using forward selection");
        System.out.println("Features: 12 available features");
        System.out.println("Classifier: Decision Tree");
        System.out.println();
        
        Part4_FeatureSelection.run();
        
        System.out.println();
        System.out.println("Part 4 completed successfully!");
        System.out.println("=" .repeat(80));
    }

    private static void runPart5() throws Exception {
        printHeader("PART 5: Classifier Comparison");
        System.out.println("Comparing Random Forest and SVM with feature selection");
        System.out.println("Features: 12 available features");
        System.out.println("Classifiers: Random Forest, SVM");
        System.out.println();
        
        Part5_ClassifierComparison.run();
        
        System.out.println();
        System.out.println("Part 5 completed successfully!");
        System.out.println("=" .repeat(80));
    }

    private static void runAll() throws Exception {
        System.out.println("=" .repeat(80));
        System.out.println("RUNNING ALL PARTS OF ASSIGNMENT 2");
        System.out.println("=" .repeat(80));
        System.out.println();
        
        runPart1();
        System.out.println();
        
        runPart2();
        System.out.println();
        
        runPart3();
        System.out.println();
        
        runPart4();
        System.out.println();
        
        runPart5();
        System.out.println();
        
        System.out.println("=" .repeat(80));
        System.out.println("ALL PARTS COMPLETED SUCCESSFULLY!");
        System.out.println("=" .repeat(80));
    }

    private static void printHeader(String title) {
        System.out.println("=" .repeat(80));
        System.out.println(title);
        System.out.println("=" .repeat(80));
    }

    private static void printUsage() {
        System.out.println("Assignment 2 Handler - Gesture Recognition Optimization");
        System.out.println();
        System.out.println("Usage:");
        System.out.println("  java -cp \".:lib/weka.jar\" Assignment2_Handler <command>");
        System.out.println();
        System.out.println("Commands:");
        System.out.println("  part1  - Run Part 1: Baseline accuracy with complete dataset");
        System.out.println("  part2  - Run Part 2: Time window tuning (1s, 2s, 3s, 4s)");
        System.out.println("  part3  - Run Part 3: Feature expansion (median & RMS)");
        System.out.println("  part4  - Run Part 4: Sequential feature selection (Decision Tree)");
        System.out.println("  part5  - Run Part 5: Classifier comparison (Random Forest & SVM)");
        System.out.println("  all    - Run all parts in sequence");
        System.out.println();
    }
}
