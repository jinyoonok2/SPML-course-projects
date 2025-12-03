// Part5_ClassifierComparison.java
// Sequential Feature Selection (SFS) for Decision Tree, Random Forest, and SVM
// Outputs all result CSV files to results/part5/

import java.nio.file.*;
import java.util.*;

public class Part5_ClassifierComparison {

    // Input features produced by Part 3 (best window length chosen)
    private static final String FEATURES_CSV_PATH = "results/part3/features.csv";
    private static final int NUM_FEATURES = 12;
    private static final double MIN_IMPROVEMENT = 0.001; // 0.1%

    private static class SFSResult {
        List<Integer> selected = new ArrayList<>();
        List<Double> accCurve = new ArrayList<>();
        double finalAcc = 0.0;
        int iterations = 0;
    }

    public static void run() throws Exception {
        // Read the feature CSV
        Path csvPath = Paths.get(FEATURES_CSV_PATH);
        if (Files.notExists(csvPath)) {
            System.err.printf("Missing features CSV at %s. Run Part 3 first.%n", csvPath.toAbsolutePath());
            return;
        }

        String[][] csvData;
        try {
            csvData = MyWekaUtils.readCSV(csvPath.toString());
            if (csvData == null || csvData.length == 0) {
                System.err.println("Features CSV was read but is empty.");
                return;
            }
        } catch (Exception e) {
            System.err.printf("Failed to load features CSV %s: %s%n", csvPath, e);
            return;
        }

        // Run SFS for three classifiers: DT (1), RF (2), SVM (3)
        SFSResult dt  = sfsForClassifier(csvData, 1, "results/part5/dt_sfs_curve.csv");
        SFSResult rf  = sfsForClassifier(csvData, 2, "results/part5/rf_sfs_curve.csv");
        SFSResult svm = sfsForClassifier(csvData, 3, "results/part5/svm_sfs_curve.csv");

        // Print summary and comparison
        System.out.println("\n=== Part 5: Classifier Comparison (SFS) ===");
        printSummary("Decision Tree", 1, dt);
        printSummary("Random Forest", 2, rf);
        printSummary("SVM", 3, svm);

        String bestName = "Decision Tree";
        double bestAcc = dt.finalAcc;
        if (rf.finalAcc > bestAcc) { bestAcc = rf.finalAcc; bestName = "Random Forest"; }
        if (svm.finalAcc > bestAcc){ bestAcc = svm.finalAcc; bestName = "SVM"; }

        System.out.printf("\n>>> Best classifier: %s (Final accuracy = %.2f%%)%n", bestName, bestAcc);
        System.out.println("Result CSV files written to folder: results/part5/");
    }

    private static void printSummary(String name, int option, SFSResult r) {
        System.out.printf("\n--- %s (option=%d) ---\n", name, option);
        System.out.println("Selected features (order): " + r.selected);
        System.out.println("Accuracy after each addition: " + r.accCurve);
        System.out.printf("Final accuracy: %.2f%%%n", r.finalAcc);
    }

    private static SFSResult sfsForClassifier(String[][] csvData, int classifierOption, String outCsvPath) throws Exception {
        System.out.printf("\n[Start SFS] Classifier option=%d%n", classifierOption);

        List<Integer> all = new ArrayList<>();
        for (int i = 0; i < NUM_FEATURES; i++) all.add(i);

        List<Integer> selected = new ArrayList<>();
        List<Double> accCurve = new ArrayList<>();
        double lastBestAcc = 0.0;

        while (selected.size() < all.size()) {
            System.out.printf("Iteration %d (selected=%s)%n", selected.size(), selected);
            int bestFeat = -1;
            double bestAcc = -1.0;

            for (int f : all) {
                if (selected.contains(f)) continue;
                int[] subset = toArrayWith(selected, f);

                String arff = MyWekaUtils.csvToArff(csvData, subset);
                double acc = MyWekaUtils.classify(arff, classifierOption);

                System.out.printf("  Try adding F%-2d → acc = %.3f%%%n", f, acc);
                if (acc > bestAcc) { bestAcc = acc; bestFeat = f; }
            }

            double improvement = bestAcc - lastBestAcc;
            if (bestFeat == -1 || improvement < MIN_IMPROVEMENT) {
                System.out.printf("Stop: improvement %.3f%% < threshold %.3f%%%n", improvement, MIN_IMPROVEMENT);
                break;
            }

            selected.add(bestFeat);
            lastBestAcc = bestAcc;
            accCurve.add(lastBestAcc);

            System.out.printf("  + Added F%d → best acc = %.3f%%%n%n", bestFeat, lastBestAcc);
        }

        saveCurve(outCsvPath, selected, accCurve);

        SFSResult r = new SFSResult();
        r.selected.addAll(selected);
        r.accCurve.addAll(accCurve);
        r.finalAcc = lastBestAcc;
        r.iterations = accCurve.size();
        System.out.printf("[End SFS] option=%d final acc=%.2f%% features=%s%n",
                classifierOption, r.finalAcc, r.selected);
        return r;
    }

    private static int[] toArrayWith(List<Integer> base, int extra) {
        int[] arr = new int[base.size() + 1];
        int i = 0;
        for (int v : base) arr[i++] = v;
        arr[i] = extra;
        Arrays.sort(arr);
        return arr;
    }

    private static void saveCurve(String outPath, List<Integer> selected, List<Double> accCurve) throws Exception {
        Path p = Paths.get(outPath);
        Files.createDirectories(p.getParent()); // ensure results/part5/ exists
        List<String> lines = new ArrayList<>();
        lines.add("k,added_feature,accuracy");
        for (int i = 0; i < accCurve.size(); i++) {
            lines.add(String.format("%d,F%d,%.4f", i + 1, selected.get(i), accCurve.get(i)));
        }
        Files.write(p, lines);
    }
}
