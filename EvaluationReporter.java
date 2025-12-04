import java.io.*;
import java.nio.file.*;
import java.text.SimpleDateFormat;
import java.util.*;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

/**
 * EvaluationReporter - Comprehensive evaluation reporting and visualization
 * 
 * Handles:
 * - Detailed text reports with all metrics
 * - Confusion matrix visualization (text and image)
 * - Performance metrics (accuracy, precision, recall, F1)
 * - Result persistence
 */
public class EvaluationReporter {
    
    /**
     * Complete evaluation result with all metrics
     */
    public static class DetailedEvaluationResult {
        public double accuracy;
        public double kappa;
        public double[][] confusionMatrix;
        public String classifierName;
        public String[] classLabels;
        public double[] precision;
        public double[] recall;
        public double[] fMeasure;
        public String fullSummary;
        public Instances instances;
        public Classifier classifier;
        
        public DetailedEvaluationResult(Evaluation eval, Classifier classifier, 
                                       String classifierName, Instances instances) {
            this.accuracy = eval.pctCorrect() / 100.0;
            this.kappa = eval.kappa();
            this.confusionMatrix = eval.confusionMatrix();
            this.classifierName = classifierName;
            this.instances = instances;
            this.classifier = classifier;
            
            // Extract class labels
            int numClasses = instances.numClasses();
            classLabels = new String[numClasses];
            for (int i = 0; i < numClasses; i++) {
                classLabels[i] = instances.classAttribute().value(i);
            }
            
            // Extract per-class metrics
            precision = new double[numClasses];
            recall = new double[numClasses];
            fMeasure = new double[numClasses];
            
            for (int i = 0; i < numClasses; i++) {
                precision[i] = eval.precision(i);
                recall[i] = eval.recall(i);
                fMeasure[i] = eval.fMeasure(i);
            }
            
            // Full summary text
            try {
                fullSummary = eval.toSummaryString("\n=== Evaluation Summary ===\n", false);
                fullSummary += "\n" + eval.toClassDetailsString();
                fullSummary += "\n=== Confusion Matrix ===\n" + eval.toMatrixString();
            } catch (Exception e) {
                fullSummary = "Error generating summary: " + e.getMessage();
            }
        }
    }
    
    /**
     * Print detailed evaluation results to console
     */
    public static void printEvaluation(DetailedEvaluationResult result) {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("EVALUATION RESULTS: " + result.classifierName);
        System.out.println("=".repeat(80));
        
        // Overall metrics
        System.out.printf("\n📊 Overall Performance:%n");
        System.out.printf("  Accuracy:  %.4f (%.2f%%)%n", result.accuracy, result.accuracy * 100);
        System.out.printf("  Kappa:     %.4f%n", result.kappa);
        
        // Per-class metrics
        System.out.printf("\n📈 Per-Class Metrics:%n");
        System.out.printf("  %-20s  Precision  Recall    F1-Score%n", "Class");
        System.out.println("  " + "-".repeat(60));
        for (int i = 0; i < result.classLabels.length; i++) {
            System.out.printf("  %-20s  %.4f     %.4f    %.4f%n", 
                result.classLabels[i], result.precision[i], result.recall[i], result.fMeasure[i]);
        }
        
        // Confusion Matrix
        System.out.printf("\n🔢 Confusion Matrix:%n");
        printConfusionMatrix(result.confusionMatrix, result.classLabels);
        
        System.out.println("=".repeat(80) + "\n");
    }
    
    /**
     * Print confusion matrix in nice format
     */
    private static void printConfusionMatrix(double[][] matrix, String[] labels) {
        int numClasses = labels.length;
        
        // Find max label length for formatting
        int maxLabelLen = 8;
        for (String label : labels) {
            maxLabelLen = Math.max(maxLabelLen, label.length());
        }
        
        // Header row
        System.out.print("  " + " ".repeat(maxLabelLen + 2));
        for (String label : labels) {
            System.out.printf("%-10s", truncate(label, 10));
        }
        System.out.println("  <-- Classified as");
        
        // Data rows
        for (int i = 0; i < numClasses; i++) {
            System.out.printf("  %-" + maxLabelLen + "s |", truncate(labels[i], maxLabelLen));
            for (int j = 0; j < numClasses; j++) {
                System.out.printf(" %-8d ", (int)matrix[i][j]);
            }
            System.out.printf(" | %s%n", labels[i]);
        }
    }
    
    /**
     * Save complete evaluation report to file
     */
    public static void saveEvaluationReport(DetailedEvaluationResult result, Path outputDir, 
                                           String experimentName) throws IOException {
        Files.createDirectories(outputDir);
        
        String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String safeClassifierName = result.classifierName.replaceAll("[^a-zA-Z0-9_-]", "_");
        String safeExperimentName = experimentName.replaceAll("[^a-zA-Z0-9_-]", "_");
        
        // Main report file
        Path reportFile = outputDir.resolve(
            String.format("%s_%s_evaluation.txt", safeExperimentName, safeClassifierName));
        
        StringBuilder report = new StringBuilder();
        report.append("=".repeat(80)).append("\n");
        report.append("EVALUATION REPORT\n");
        report.append("=".repeat(80)).append("\n\n");
        report.append("Experiment: ").append(experimentName).append("\n");
        report.append("Classifier: ").append(result.classifierName).append("\n");
        report.append("Timestamp:  ").append(timestamp).append("\n");
        report.append("Instances:  ").append(result.instances.numInstances()).append("\n");
        report.append("Features:   ").append(result.instances.numAttributes() - 1).append("\n");
        report.append("Classes:    ").append(result.instances.numClasses()).append("\n\n");
        
        // Overall metrics
        report.append("OVERALL PERFORMANCE\n");
        report.append("-".repeat(80)).append("\n");
        report.append(String.format("Accuracy:  %.4f (%.2f%%)%n", result.accuracy, result.accuracy * 100));
        report.append(String.format("Kappa:     %.4f%n%n", result.kappa));
        
        // Per-class metrics
        report.append("PER-CLASS METRICS\n");
        report.append("-".repeat(80)).append("\n");
        report.append(String.format("%-20s  Precision  Recall    F1-Score%n", "Class"));
        report.append("-".repeat(80)).append("\n");
        for (int i = 0; i < result.classLabels.length; i++) {
            report.append(String.format("%-20s  %.4f     %.4f    %.4f%n", 
                result.classLabels[i], result.precision[i], result.recall[i], result.fMeasure[i]));
        }
        report.append("\n");
        
        // Confusion Matrix (text)
        report.append("CONFUSION MATRIX\n");
        report.append("-".repeat(80)).append("\n");
        report.append(getConfusionMatrixText(result.confusionMatrix, result.classLabels));
        report.append("\n\n");
        
        // Full Weka summary
        report.append("DETAILED WEKA OUTPUT\n");
        report.append("-".repeat(80)).append("\n");
        report.append(result.fullSummary).append("\n");
        
        Files.write(reportFile, report.toString().getBytes());
        System.out.printf("📄 Evaluation report saved: %s%n", reportFile);
        
        // Save confusion matrix as CSV for plotting
        saveConfusionMatrixCSV(result, outputDir, safeExperimentName, safeClassifierName);
    }
    
    /**
     * Get confusion matrix as formatted text
     */
    private static String getConfusionMatrixText(double[][] matrix, String[] labels) {
        StringBuilder sb = new StringBuilder();
        int numClasses = labels.length;
        
        // Find max label length
        int maxLabelLen = 8;
        for (String label : labels) {
            maxLabelLen = Math.max(maxLabelLen, label.length());
        }
        
        // Header
        sb.append(" ".repeat(maxLabelLen + 2));
        for (String label : labels) {
            sb.append(String.format("%-10s", truncate(label, 10)));
        }
        sb.append(" <-- Classified as\n");
        
        // Rows
        for (int i = 0; i < numClasses; i++) {
            sb.append(String.format("%-" + maxLabelLen + "s |", truncate(labels[i], maxLabelLen)));
            for (int j = 0; j < numClasses; j++) {
                sb.append(String.format(" %-8d ", (int)matrix[i][j]));
            }
            sb.append(String.format(" | %s%n", labels[i]));
        }
        
        return sb.toString();
    }
    
    /**
     * Save confusion matrix as CSV file (for plotting)
     */
    private static void saveConfusionMatrixCSV(DetailedEvaluationResult result, Path outputDir,
                                              String experimentName, String classifierName) throws IOException {
        Path csvFile = outputDir.resolve(
            String.format("%s_%s_confusion_matrix.csv", experimentName, classifierName));
        
        StringBuilder csv = new StringBuilder();
        
        // Header row
        csv.append("Actual/Predicted");
        for (String label : result.classLabels) {
            csv.append(",").append(label);
        }
        csv.append("\n");
        
        // Data rows
        for (int i = 0; i < result.classLabels.length; i++) {
            csv.append(result.classLabels[i]);
            for (int j = 0; j < result.classLabels.length; j++) {
                csv.append(",").append((int)result.confusionMatrix[i][j]);
            }
            csv.append("\n");
        }
        
        Files.write(csvFile, csv.toString().getBytes());
        System.out.printf("📊 Confusion matrix CSV saved: %s%n", csvFile);
    }
    
    /**
     * Truncate string to max length
     */
    private static String truncate(String str, int maxLen) {
        if (str.length() <= maxLen) return str;
        return str.substring(0, maxLen - 2) + "..";
    }
}
