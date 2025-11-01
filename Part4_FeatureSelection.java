/**
 * Part4_FeatureSelection
 * ----------------------
 * Part 4: Sequential feature selection with Decision Tree
 * 
 * TODO: Implementation coming soon
 */

import java.nio.file.*;
import java.util.*;
import static java.util.stream.Collectors.*;

public class Part4_FeatureSelection {

    // === Configure these ===
    // Build this CSV using the BEST window from Part 2 before running Part 4.
    private static final String FEATURES_CSV_PATH = "results/part3/features.csv";

    private static final int NUM_FEATURES = 12; 

    // Stop when the best next addition improves accuracy by less than this (absolute %)
    private static final double MIN_IMPROVEMENT = 0.001; // 1%

    public static void run() throws Exception {
        // 1) Read CSV once (do exactly as in Part 2)

        Path csvPath = Paths.get(FEATURES_CSV_PATH);
        String[][] csvData = null;

        try {
            if (Files.notExists(csvPath)) {
                System.err.printf("Missing features CSV at %s. Did you run Part 3 to generate it?%n",
                          csvPath.toAbsolutePath());
                return; 
            }
            csvData = MyWekaUtils.readCSV(csvPath.toString());

            if (csvData == null || csvData.length == 0) {
                System.err.println("Features CSV was read but is empty.");
                return; 
            }
        } catch (java.io.IOException e) {
            System.err.printf("I/O error while reading %s: %s%n", csvPath, e.getMessage());
            return;
        } catch (Exception e) {
            // If MyWekaUtils.readCSV throws something else (e.g., parsing error)
            System.err.printf("Failed to load features CSV %s: %s%n", csvPath, e.toString());
            return;
        }

        // 2) Prepare candidate feature indices [0..NUM_FEATURES-1]
        List<Integer> all = new ArrayList<>();
        for (int i = 0; i < NUM_FEATURES; i++) all.add(i);

        // 3) Sequential Forward Selection
        List<Integer> selected = new ArrayList<>();
        double lastBestAcc = 0.0;

        // For reporting: accuracy after each chosen feature
        List<Double> accCurve = new ArrayList<>();

        while (selected.size() < all.size()) {
            System.out.printf("Iteration: %d%n", selected.size());
            int bestFeatToAdd = -1;
            double bestAccThisRound = -1.0;

            // Try each remaining feature
            for (int f : all) {
                if (selected.contains(f)) continue;

                int[] trialSubset = toArrayWith(selected, f);
                // Convert exactly these features → ARFF (DO NOT include class index) :contentReference[oaicite:6]{index=6}
                String arff = MyWekaUtils.csvToArff(csvData, trialSubset);

                // Evaluate with Decision Tree (option 1) :contentReference[oaicite:7]{index=7}
                double acc = MyWekaUtils.classify(arff, 1);
                System.out.printf("Trying feature: F%d ", f);
                System.out.printf("Resulting acc: %.3f%n", acc);

                if (acc > bestAccThisRound) {
                    bestAccThisRound = acc;
                    bestFeatToAdd = f;
                }
            }

            // Check improvement threshold (absolute %)
            double improvement = bestAccThisRound - lastBestAcc;
            if (bestFeatToAdd == -1 || improvement < MIN_IMPROVEMENT) {
                // Stop: no meaningful gain or nothing left
                System.out.printf("The improvement of this iteration does not meet the threshold: %n" +
                "minimum improvement threadhold=%.3f, actual improvement=%.3f; bestFeatToAdd=F%d. %n" +
                "Feature Selection stops!%n", MIN_IMPROVEMENT, improvement, bestFeatToAdd);
                break;
            }

            // Commit the winner
            selected.add(bestFeatToAdd);
            lastBestAcc = bestAccThisRound;
            accCurve.add(lastBestAcc);

            System.out.printf("Added feature F%d → accuracy = %.3f%%%n",
                    bestFeatToAdd, lastBestAcc);
            System.out.println("Current selected features: " + selected + "\n\n");

        }

        // Final report
        System.out.println("\n=== Part 4 Result (Decision Tree) ===");
        System.out.println("Selected features (by index): " + selected);
        System.out.println("Accuracy after each addition: " + accCurve);
        System.out.printf("Final accuracy: %.2f%%%n", lastBestAcc);

        // (Optional) save a small CSV log to include in your report
        saveCurve("results/part4_sfs_curve.csv", selected, accCurve);
    }

    private static int[] toArrayWith(List<Integer> base, int extra) {
        int[] arr = new int[base.size() + 1];
        int i = 0;
        for (int v : base) arr[i++] = v;
        arr[i] = extra;
        Arrays.sort(arr); // keep indices sorted (nice to have)
        return arr;
    }

    private static void saveCurve(String outPath, List<Integer> selected, List<Double> accCurve) throws Exception {
        Path p = Paths.get(outPath);
        Files.createDirectories(p.getParent());
        List<String> lines = new ArrayList<>();
        lines.add("k,added_feature,accuracy");
        double prev = 0;
        for (int i = 0; i < accCurve.size(); i++) {
            int f = selected.get(i);
            double acc = accCurve.get(i);
            lines.add(String.format("%d,F%d,%.4f", i + 1, f, acc));
            prev = acc;
        }
        Files.write(p, lines);
    }
}
