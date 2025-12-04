# SPML Course - Activity Recognition System

## Overview
A modular machine learning system for activity recognition using smartwatch accelerometer data. Implements a complete optimization pipeline from raw sensor data to optimized classification models.

**Current Application**: Dumbbell Exercise Classification (bicep_curl, hammer_curl, tricep_kickback)
- Using 50Hz accelerometer data from left wrist
- Distinguishes between easily confused exercises with similar movement patterns

**Key Features**: 
- **Multi-class support** - Automatically detects any number of activities from filenames
- **Per-classifier optimization** - Each classifier gets its own optimal window size and feature set
- **Modular architecture** - 7 specialized components with clean separation of concerns
- **Automatic logging** - All console output saved to log files
- **Comprehensive evaluation** - Detailed metrics, confusion matrices, per-class statistics
- **Reproducible experiments** - Complete pipeline with numbered step structure (1→2→3→4)

## Project Structure

```
SPML-course-projects/
├── raw_data/                           # Raw CSV files from smartwatch (gitignored)
│   └── WatchID-FinalProject-Subject-left-Activity-Info-DateTime.csv
│       # Activity: bicep_curl, hammer_curl, tricep_kickback
│       # Examples: Watch123-FinalProject-Jinyoon-left-bicep_curl-Info-Date.csv
│
├── formatted_data/                     # Cleaned data (timestamp, ax, ay, az) (gitignored)
│   └── formatted_WatchID-FinalProject-Subject-left-Activity.csv
│
├── results/                            # All experiment outputs (gitignored - generated locally)
│   ├── 1_baseline/                     # Step 1: Initial baseline evaluation
│   │   ├── baseline_log.txt            # Complete console output from baseline run
│   │   ├── j48/
│   │   │   ├── Baseline_J48_evaluation_report.txt
│   │   │   ├── Baseline_J48_confusion_matrix.csv
│   │   │   └── Baseline_J48_confusion_matrix.png
│   │   ├── random_forest/
│   │   │   ├── Baseline_RandomForest_evaluation_report.txt
│   │   │   ├── Baseline_RandomForest_confusion_matrix.csv
│   │   │   └── Baseline_RandomForest_confusion_matrix.png
│   │   └── smo/
│   │       ├── Baseline_SMO_evaluation_report.txt
│   │       ├── Baseline_SMO_confusion_matrix.csv
│   │       └── Baseline_SMO_confusion_matrix.png
│   │
│   ├── 2_window_optimization/          # Step 2: Window size tuning
│   │   ├── window_optimization_log.txt # Complete console output from optimization
│   │   ├── j48/
│   │   │   ├── WindowOpt_J48_1.0s_evaluation_report.txt
│   │   │   ├── WindowOpt_J48_1.0s_confusion_matrix.csv
│   │   │   ├── WindowOpt_J48_2.0s_evaluation_report.txt
│   │   │   ├── ... (reports for all tested window sizes)
│   │   │   └── WindowOpt_J48_confusion_matrix.png (best window)
│   │   ├── random_forest/
│   │   │   └── ... (same structure)
│   │   └── smo/
│   │       └── ... (same structure)
│   │
│   ├── 3_feature_expansion/            # Step 3: Master datasets with expanded features
│   │   ├── feature_expansion_log.txt   # Complete console output from feature expansion
│   │   ├── j48_master_dataset_12features.csv
│   │   ├── random_forest_master_dataset_12features.csv
│   │   └── smo_master_dataset_12features.csv
│   │
│   └── 4_feature_selection/            # Step 4: Sequential Feature Selection
│       ├── feature_selection_log.txt   # Complete console output from SFS
│       ├── sfs_j48/
│       │   ├── SFS_J48_7features.csv           # Dataset with selected features
│       │   ├── SFS_J48_7features_evaluation_report.txt
│       │   ├── SFS_J48_7features_confusion_matrix.csv
│       │   └── SFS_J48_7features_confusion_matrix.png
│       ├── sfs_random_forest/
│       │   ├── SFS_RandomForest_8features.csv
│       │   ├── SFS_RandomForest_8features_evaluation_report.txt
│       │   ├── SFS_RandomForest_8features_confusion_matrix.csv
│       │   └── SFS_RandomForest_8features_confusion_matrix.png
│       └── sfs_smo/
│           ├── SFS_SMO_6features.csv
│           ├── SFS_SMO_6features_evaluation_report.txt
│           ├── SFS_SMO_6features_confusion_matrix.csv
│           └── SFS_SMO_6features_confusion_matrix.png
│
├── legacy/                             # Archived old Part1-5 files (for reference only)
│   ├── Part1_DataProcessor.java
│   ├── Part2_WindowTuning.java
│   ├── Part3_FeatureExpansion.java
│   ├── Part4_FeatureSelection.java
│   ├── Part5_ClassifierComparison.java
│   ├── WadaManager_old.java
│   └── Assignment2_Handler.java
│
├── lib/
│   └── weka.jar                        # Weka ML library 3.8.6 (gitignored - download separately)
│
├── config.json                         # Project configuration
├── .gitignore                          # Git exclusion rules
├── check_jdk.sh                        # JDK verification script
├── compile.sh                          # Compilation script (with --clean option)
├── environment.yml                     # Conda environment for Python visualization
├── plot_confusion_matrix.py            # Generate confusion matrix PNG images
│
├── ExperimentRunner.java               # Main entry point and orchestration (482 lines)
├── DataManager.java                    # Data processing engine (244 lines)
├── FeatureEngine.java                  # Feature extraction engine (243 lines)
├── OptimizationEngine.java             # Window optimization and SFS engine (382 lines)
├── MLEngine.java                       # Machine learning operations engine (263 lines)
├── EvaluationReporter.java             # Comprehensive evaluation reporting (263 lines)
├── ExperimentLogger.java               # Automatic dual-stream logging (89 lines)
├── MyWekaUtils.java                    # Core Weka utilities (112 lines)
│
└── README.md                           # This file
```

**File Categories:**

📁 **Gitignored** (generated locally, not in version control):
- `raw_data/`, `formatted_data/`, `results/` - Data and outputs
- `lib/weka.jar` - Download separately
- `*.class`, `*.arff` - Compiled files

✅ **Version Controlled** (in Git):
- All `.java` source files
- Configuration files (`config.json`, `environment.yml`)
- Scripts (`.sh`, `.py`)
- Documentation

**Data Sharing**: Raw data available via Google Drive link (see below). Download, unzip to `raw_data/`, and run experiments to reproduce results.


## Code Architecture

### System Components (7 Files)

#### 1. **ExperimentRunner.java** (482 lines) - Main Entry Point
**Purpose**: Orchestrates all experiments and provides command-line interface

**Key Responsibilities**:
- Parse command-line arguments and dispatch to appropriate workflows
- Run complete experiment pipeline (baseline → optimization → expansion → selection)
- Coordinate data processing, feature engineering, and ML operations
- Manage experiment directories and ensure proper sequencing

**Key Methods**:
- `main(String[] args)`: Entry point, command parsing
- `runCompleteExperiment()`: Executes full 4-step pipeline with automatic logging
- `runBaseline()`: Step 1 - Baseline evaluation (6 features, 1s window, all classifiers)
- `runWindowOptimization()`: Step 2 - Find optimal window size per classifier
- `runFeatureExpansion()`: Step 3 - Create master datasets with 12 features
- `runFeatureSelection()`: Step 4 - SFS to find optimal feature subset per classifier
- `runCustomExperiment(...)`: Flexible experiment with custom parameters

**Automatic Logging**: Integrates ExperimentLogger for steps 3 and 4 to save console output

**Dependencies**: Uses DataManager, FeatureEngine, OptimizationEngine, MLEngine

---

#### 2. **DataManager.java** (244 lines) - Data Processing Engine
**Purpose**: Handle all data format conversions and file management

**Key Responsibilities**:
- Format raw CSV files (remove sensor_type and accuracy columns)
- Convert formatted CSV to Weka ARFF format
- Extract activity labels from filenames (supports any activity name)
- Automatic multi-class detection from unique activities in directory
- Manage data directories and file paths

**Key Methods**:
- `formatAllData()`: Batch process all files in raw_data/, output to formatted_data/
- `formatData(Path rawFile)`: Format single CSV (keep timestamp, ax, ay, az)
- `extractActivityFromFilename(String filename)`: Parse activity label from filename
- `csvToArff(Path csvPath)`: Convert CSV features file to ARFF for Weka
- `getUniqueActivities(Path directory)`: Detect all activity classes from filenames

**File Naming Convention**: 
- Input: `WatchID-Project-Subject-Placement-Activity-Info-DateTime.csv`
- Output: `formatted_WatchID-Project-Subject-Placement-Activity.csv`
- Activity extraction: Split by `-`, find activity field (e.g., `bicep_curl`)

**Multi-class Support**: Automatically discovers all unique activities - no hardcoding required

---

#### 3. **FeatureEngine.java** (243 lines) - Feature Extraction Engine
**Purpose**: Extract statistical features from time-series accelerometer data

**Key Responsibilities**:
- Sliding window feature extraction from formatted CSVs
- Support both basic (6) and expanded (12) feature sets
- Handle variable window sizes (1s - 4s)
- Preserve activity labels through extraction process

**Feature Sets**:

**Basic Features (6)**:
- `mean_x`, `mean_y`, `mean_z` - Average acceleration per axis
- `std_x`, `std_y`, `std_z` - Standard deviation per axis

**Expanded Features (12)**:
- `mean_x`, `mean_y`, `mean_z` - Average acceleration
- `std_x`, `std_y`, `std_z` - Standard deviation (variability)
- `median_x`, `median_y`, `median_z` - Median values (robust to outliers)
- `rms_x`, `rms_y`, `rms_z` - Root Mean Square (signal magnitude)

**Key Methods**:
- `extractBasicFeatures(Path formattedFile, int windowMs)`: Extract 6 features
- `extractExpandedFeatures(Path formattedFile, int windowMs)`: Extract 12 features
- `calculateBasicFeatures(List<DataPoint> window)`: Compute 6 statistics
- `calculateExpandedFeatures(List<DataPoint> window)`: Compute 12 statistics
- `extractAllFeatures(...)`: Process entire directory with sliding windows

**Parameters**:
- `windowMs`: Window size in milliseconds (e.g., 1000 = 1 second)
- Stride: 1 second (1000ms) for multi-second windows
- Sampling rate: 50Hz (configured in code)

**Output Format**: CSV with feature columns + Activity column

---

#### 4. **OptimizationEngine.java** (382 lines) - Optimization and Feature Selection
**Purpose**: Find optimal parameters (window size, features) for each classifier

**Key Responsibilities**:
- Window size optimization (test 1s, 2s, 3s, 4s per classifier)
- Sequential Feature Selection (SFS) using forward selection
- Save/load optimization configurations
- Generate detailed evaluation reports for each configuration

**Key Methods**:

**Window Optimization**:
- `findOptimalWindowSize(Path formattedDir, String classifier, Path resultsDir)`: 
  - Tests window sizes: 1000ms, 2000ms, 3000ms, 4000ms
  - Evaluates each with basic 6 features
  - Returns optimal window size per classifier
  - Saves detailed reports for each window size
  - Uses ExperimentLogger for automatic logging

**Sequential Feature Selection (SFS)**:
- `performSequentialFeatureSelection(Path csvFile, String classifier, Path outputDir)`:
  - Primary SFS method with explicit output directory
  - Used by Step 4 (feature selection)
  
- `performSequentialFeatureSelection(Path csvFile, String classifier)`:
  - Backward-compatible wrapper
  - Uses csvFile.getParent() as output directory
  
- `performSFSCore(Instances csvData, String classifier)`:
  - PRIVATE: Core SFS algorithm logic
  - Forward selection: Start empty, add best feature iteratively
  - Stopping criterion: Improvement < MIN_IMPROVEMENT (0.001)
  - Returns list of selected feature indices
  
- `saveSFSResults(Instances csvData, List<Integer> selectedFeatures, Path sfsReportDir, String classifier)`:
  - PRIVATE: Save SFS results to files
  - Creates CSV with only selected features (class column LAST)
  - Generates evaluation report and confusion matrix
  - **Critical**: Class column must be LAST for Weka compatibility

**Configuration**:
- `MIN_IMPROVEMENT = 0.001`: SFS stopping threshold (0.1% improvement)
- `CROSS_VALIDATION_FOLDS = 10`: 10-fold cross-validation

**Output Files**:
- Window optimization: One report per window size per classifier
- SFS: Dataset with selected features + evaluation reports

**Important Bug Fix**: SFS CSV format now correctly places class column LAST (was first, caused Weka parse errors)

---

#### 5. **MLEngine.java** (263 lines) - Machine Learning Operations
**Purpose**: Execute ML workflows and evaluations using Weka

**Key Responsibilities**:
- Baseline evaluation with all classifiers
- Complete experiment pipeline coordination
- Classifier evaluation with detailed metrics
- Integration with EvaluationReporter for comprehensive reporting

**Key Methods**:

**Baseline Evaluation**:
- `runBaselineEvaluation(Path formattedDir)`:
  - Creates baseline features: 6 features, 1s window
  - Evaluates all 3 classifiers (J48, RandomForest, SVM)
  - Saves results to `results/1_baseline/`
  - Uses ExperimentLogger to save `baseline_log.txt`
  - Generates evaluation reports and confusion matrices

**Complete Pipeline**:
- `runCompleteExperiment(Path formattedDir, Path resultsDir)`:
  - Orchestrates full 4-step workflow
  - Step 1: Baseline evaluation
  - Step 2: Window optimization (calls OptimizationEngine)
  - Step 3: Feature expansion with optimal windows
  - Step 4: SFS on expanded features
  - Ensures proper directory creation
  - Passes correct output directories to SFS

**Evaluation**:
- `evaluateClassifier(String classifier, Path arffFile)`:
  - Simple evaluation, returns accuracy
  
- `evaluateClassifier(String classifier, Path arffFile, Path outputDir)`:
  - Detailed evaluation with reporting
  - Generates evaluation report (text file)
  - Saves confusion matrix (CSV)
  - Prints to console
  - Returns DetailedEvaluationResult object

**Supported Classifiers**:
- `J48`: Weka's C4.5 decision tree
- `RandomForest`: Ensemble of decision trees
- `SVM` (SMO): Support Vector Machine

**Configuration**:
- `CROSS_VALIDATION_FOLDS = 10`: 10-fold cross-validation for all evaluations

**Integration**: Works closely with EvaluationReporter for detailed metrics

---

#### 6. **EvaluationReporter.java** (263 lines) - Comprehensive Evaluation Reporting
**Purpose**: Generate detailed evaluation metrics and save reports

**Key Responsibilities**:
- Extract comprehensive metrics from Weka Evaluation object
- Format and print evaluation results to console
- Save detailed text reports to files
- Export confusion matrices as CSV for visualization
- Calculate per-class precision, recall, F1-score

**Data Structures**:

**DetailedEvaluationResult** (inner class):
- `accuracy`: Overall classification accuracy
- `kappa`: Cohen's Kappa statistic (agreement beyond chance)
- `confusionMatrix`: 2D array of classification results
- `classLabels`: Activity names (bicep_curl, hammer_curl, etc.)
- `precision`: Per-class precision (positive predictive value)
- `recall`: Per-class recall (sensitivity, true positive rate)
- `f1Score`: Per-class F1-score (harmonic mean of precision/recall)

**Key Methods**:

**Extract Metrics**:
- `extractEvaluationResults(Evaluation eval, Instances data, String experimentName)`:
  - Parses Weka Evaluation object
  - Extracts all relevant metrics
  - Returns DetailedEvaluationResult with complete information

**Console Output**:
- `printEvaluation(DetailedEvaluationResult result)`:
  - Formatted console output with:
    - Accuracy and Kappa statistic
    - Complete confusion matrix with row/column labels
    - Per-class metrics table (Precision, Recall, F1-Score)
  - Clean, readable format for quick analysis

**File Output**:
- `saveEvaluationReport(DetailedEvaluationResult result, Path outputPath)`:
  - Complete text report with timestamp
  - All metrics in human-readable format
  - Full Weka summary included
  - Filename: `<experimentName>_evaluation_report.txt`

- `saveConfusionMatrixCSV(DetailedEvaluationResult result, Path outputPath)`:
  - CSV format with class labels as headers
  - Ready for visualization tools
  - Can be processed by plot_confusion_matrix.py
  - Filename: `<experimentName>_confusion_matrix.csv`

**Output Format Example** (Confusion Matrix CSV):
```csv
,bicep_curl,hammer_curl,tricep_kickback
bicep_curl,145,3,2
hammer_curl,2,148,0
tricep_kickback,1,0,149
```

**Usage**: Called by MLEngine whenever detailed evaluation is needed

---

#### 7. **ExperimentLogger.java** (89 lines) - Automatic Dual-Stream Logging
**Purpose**: Capture all console output to both terminal and log files simultaneously

**Key Responsibilities**:
- Redirect System.out to write to both console and file
- Maintain original console output (user sees everything)
- Add timestamps to log files
- Ensure proper cleanup with guaranteed stream closure
- Preserve original System.out after logging stops

**Key Methods**:

- `startLogging(Path logFile)`:
  - Saves original System.out
  - Creates new PrintStream that writes to both console and file
  - Replaces System.out with dual-stream
  - Writes header with timestamp to log file
  - Creates parent directories if needed

- `stopLogging()`:
  - Writes footer with timestamp
  - Closes file stream
  - Restores original System.out
  - Safe to call multiple times (idempotent)

**Usage Pattern** (try-finally for guaranteed cleanup):
```java
ExperimentLogger logger = new ExperimentLogger();
try {
    logger.startLogging(logPath);
    // ... experiment code ...
    // All System.out.println() calls are logged
} finally {
    logger.stopLogging();  // Always executes
}
```

**Log File Format**:
```
========================================
Experiment Log
Started: 2025-12-04 15:30:45
========================================

[All console output here]

========================================
Log ended: 2025-12-04 15:45:23
========================================
```

**Integration Points**:
- MLEngine.runBaselineEvaluation() → `baseline_log.txt`
- OptimizationEngine.findOptimalWindowSize() → `window_optimization_log.txt`
- ExperimentRunner.runFeatureExpansion() → `feature_expansion_log.txt`
- ExperimentRunner.runFeatureSelection() → `feature_selection_log.txt`

**Benefits**:
- Complete record of all experiments
- Reproduce exact console output later
- Share results without re-running experiments
- Track experiment history and debugging

---

#### Supporting Files

**MyWekaUtils.java** (112 lines) - Core Weka Utilities
- `buildClassifier(String classifier, Path arffFile)`: Train classifier
- `classifyWithDetails(String classifier, Path arffFile)`: Evaluate with full metrics
- `getInstancesFromArff(Path arffFile)`: Load ARFF into Weka Instances
- `getClassifierName(String shortName)`: Map short names to full class names

**config.json** - Project Configuration
```json
{
  "project": "FinalProject",
  "activities": ["bicep_curl", "hammer_curl", "tricep_kickback"],
  "subjects": ["Jinyoon", "Josh", "Katz"],
  "placements": ["left"]
}
```

**environment.yml** - Python Visualization Environment
- Conda environment: `spml-viz`
- Python 3.10
- Dependencies: numpy, matplotlib, seaborn, pandas
- For generating confusion matrix PNG images

**plot_confusion_matrix.py** (95 lines) - Confusion Matrix Visualization
- Reads CSV confusion matrices
- Generates PNG heatmap images using seaborn
- 300 DPI high-quality output
- Supports single file or batch directory processing


## Experiment Workflow

### Complete 4-Step Pipeline

The system implements a systematic optimization workflow where each classifier (J48, RandomForest, SVM) is independently optimized:

#### **Step 1: Baseline Evaluation** (`results/1_baseline/`)

**Purpose**: Establish initial performance with minimal features

**Configuration**:
- Features: 6 basic features (mean, std per axis)
- Window size: 1 second (1000ms)
- Classifiers: All three (J48, RandomForest, SVM)
- Evaluation: 10-fold cross-validation

**What Happens**:
1. Format all raw CSV files in `raw_data/`
2. Extract 6 features with 1s sliding window
3. Convert to ARFF format
4. Evaluate each classifier independently
5. Generate detailed reports for each classifier

**Output Files** (per classifier):
```
results/1_baseline/
├── baseline_log.txt                          # Complete console output
├── j48/
│   ├── Baseline_J48_evaluation_report.txt    # Detailed metrics
│   ├── Baseline_J48_confusion_matrix.csv     # For visualization
│   └── Baseline_J48_confusion_matrix.png     # (generated by Python script)
├── random_forest/
│   └── ... (same structure)
└── smo/
    └── ... (same structure)
```

**Expected Results**: Baseline accuracies to compare against optimized versions

**Command**: `java -cp ".:lib/weka.jar" ExperimentRunner baseline`

---

#### **Step 2: Window Optimization** (`results/2_window_optimization/`)

**Purpose**: Find optimal time window size for each classifier

**Configuration**:
- Features: 6 basic features (consistent with baseline)
- Window sizes tested: 1s, 2s, 3s, 4s (1000ms, 2000ms, 3000ms, 4000ms)
- Classifiers: Each tested independently
- Evaluation: 10-fold cross-validation per window size

**What Happens**:
1. For each classifier:
   - Test with 1s window → evaluate
   - Test with 2s window → evaluate
   - Test with 3s window → evaluate
   - Test with 4s window → evaluate
2. Compare accuracies across all window sizes
3. Select window with highest accuracy per classifier
4. Save optimal window configuration

**Why Different Windows Matter**:
- 1s: Captures quick movements, high temporal resolution
- 2s: Better for repetitive patterns
- 3s: Smooths out noise, captures longer movements
- 4s: Best for slow, deliberate exercises

**Output Files** (per classifier per window):
```
results/2_window_optimization/
├── window_optimization_log.txt                # Complete console output
├── j48/
│   ├── WindowOpt_J48_1.0s_evaluation_report.txt
│   ├── WindowOpt_J48_1.0s_confusion_matrix.csv
│   ├── WindowOpt_J48_2.0s_evaluation_report.txt
│   ├── WindowOpt_J48_2.0s_confusion_matrix.csv
│   ├── WindowOpt_J48_3.0s_evaluation_report.txt
│   ├── WindowOpt_J48_3.0s_confusion_matrix.csv
│   ├── WindowOpt_J48_4.0s_evaluation_report.txt
│   ├── WindowOpt_J48_4.0s_confusion_matrix.csv
│   └── WindowOpt_J48_confusion_matrix.png    # Best window only
├── random_forest/
│   └── ... (same structure)
└── smo/
    └── ... (same structure)
```

**Expected Results**: Each classifier may prefer different window sizes

**Example Output**:
```
J48: Optimal window = 2.0s (Accuracy: 94.5%)
RandomForest: Optimal window = 3.0s (Accuracy: 96.2%)
SVM: Optimal window = 2.0s (Accuracy: 91.8%)
```

**Command**: `java -cp ".:lib/weka.jar" ExperimentRunner optimize`

---

#### **Step 3: Feature Expansion** (`results/3_feature_expansion/`)

**Purpose**: Create master datasets with expanded features using optimal windows

**Configuration**:
- Features: 12 expanded features (mean, std, median, RMS per axis)
- Window size: Each classifier uses its optimal window from Step 2
- Classifiers: Each gets its own master dataset
- Output: CSV files for SFS in Step 4

**What Happens**:
1. For each classifier:
   - Load its optimal window size from Step 2
   - Extract 12 features (instead of 6) from formatted data
   - Use the classifier's optimal window size
   - Save as master dataset CSV

**Feature Expansion Details**:
- **Basic 6**: mean_x, std_x, mean_y, std_y, mean_z, std_z
- **Added 6**: median_x, median_y, median_z, rms_x, rms_y, rms_z
- **Total**: 12 features + Activity class label

**Why Expanded Features**:
- Median: Robust to outliers, captures central tendency
- RMS: Signal magnitude, captures overall movement intensity
- More features → More information for SFS to choose from

**Output Files**:
```
results/3_feature_expansion/
├── feature_expansion_log.txt                   # Complete console output
├── j48_master_dataset_12features.csv           # J48's dataset (2s window)
├── random_forest_master_dataset_12features.csv # RF's dataset (3s window)
└── smo_master_dataset_12features.csv           # SVM's dataset (2s window)
```

**CSV Format** (12 features + class):
```csv
mean_x,std_x,median_x,rms_x,mean_y,std_y,median_y,rms_y,mean_z,std_z,median_z,rms_z,Activity
-4.52,0.42,-4.51,4.54,1.23,0.57,1.21,1.36,8.90,0.35,8.89,8.91,bicep_curl
-2.13,0.82,-2.15,2.29,0.46,0.23,0.45,0.51,9.23,0.57,9.21,9.25,hammer_curl
...
```

**Expected Results**: 3 master datasets ready for feature selection

**Command**: `java -cp ".:lib/weka.jar" ExperimentRunner features`

---

#### **Step 4: Sequential Feature Selection** (`results/4_feature_selection/`)

**Purpose**: Select optimal subset of features for each classifier using forward SFS

**Configuration**:
- Input: Master datasets from Step 3 (12 features each)
- Algorithm: Forward Sequential Feature Selection
- Stopping criterion: Improvement < 0.001 (0.1%)
- Classifiers: Each selects from its own master dataset
- Evaluation: 10-fold cross-validation

**How SFS Works** (Forward Selection):
1. Start with empty feature set
2. Try adding each unused feature individually
3. Evaluate accuracy with that feature added
4. Keep the feature that gives highest improvement
5. Repeat steps 2-4 until improvement < MIN_IMPROVEMENT (0.001)
6. Return selected features

**Why SFS Matters**:
- Reduces overfitting by eliminating redundant features
- Improves interpretability (fewer features to analyze)
- Can improve accuracy by removing noisy features
- Each classifier may prefer different feature subsets

**What Happens**:
1. For each classifier:
   - Load its master dataset (12 features)
   - Run forward SFS algorithm
   - Select optimal feature subset (typically 5-10 features)
   - Create new CSV with only selected features
   - Evaluate final model with selected features
   - Save results

**Output Files** (per classifier):
```
results/4_feature_selection/
├── feature_selection_log.txt                     # Complete console output
├── sfs_j48/
│   ├── SFS_J48_7features.csv                     # Dataset with 7 selected features
│   ├── SFS_J48_7features_evaluation_report.txt   # Final evaluation metrics
│   ├── SFS_J48_7features_confusion_matrix.csv    # Confusion matrix
│   └── SFS_J48_7features_confusion_matrix.png    # (generated by Python script)
├── sfs_random_forest/
│   ├── SFS_RandomForest_8features.csv
│   ├── SFS_RandomForest_8features_evaluation_report.txt
│   ├── SFS_RandomForest_8features_confusion_matrix.csv
│   └── SFS_RandomForest_8features_confusion_matrix.png
└── sfs_smo/
    ├── SFS_SMO_6features.csv
    ├── SFS_SMO_6features_evaluation_report.txt
    ├── SFS_SMO_6features_confusion_matrix.csv
    └── SFS_SMO_6features_confusion_matrix.png
```

**Example SFS Progress** (printed during execution):
```
Starting SFS for J48 with 12 features...
Current features: [] | Accuracy: 0.0000
Adding feature: mean_x | Accuracy: 0.7850 | Improvement: 0.7850
Adding feature: std_z | Accuracy: 0.8920 | Improvement: 0.1070
Adding feature: rms_y | Accuracy: 0.9450 | Improvement: 0.0530
Adding feature: median_x | Accuracy: 0.9650 | Improvement: 0.0200
Adding feature: std_y | Accuracy: 0.9720 | Improvement: 0.0070
Adding feature: mean_z | Accuracy: 0.9760 | Improvement: 0.0040
Adding feature: rms_x | Accuracy: 0.9775 | Improvement: 0.0015
Adding feature: median_z | Accuracy: 0.9782 | Improvement: 0.0007 ✗ (< 0.001)
SFS Complete: Selected 7 features
Final Accuracy: 0.9775 (97.75%)
```

**Expected Results**: 
- Final optimized models with best features
- Typically 5-10 features selected per classifier
- Highest accuracies achieved (usually improves over baseline)

**Important**: These SFS results ARE the final comparison - no separate Step 5 needed

**Command**: `java -cp ".:lib/weka.jar" ExperimentRunner selection`

---

### Numbered Directory Structure

Results are organized in **numbered directories** (1-4) to show execution order:

1. **`1_baseline/`** - Start here: Initial performance reference
2. **`2_window_optimization/`** - Next: Find best time window
3. **`3_feature_expansion/`** - Then: Create rich master datasets
4. **`4_feature_selection/`** - Finally: Select optimal features

**Why Numbered**:
- Clear execution sequence
- Easy to track progress
- Logical workflow visualization
- Prevents confusion about step order

**What Each Step Produces**:
- **Step 1**: Evaluation reports (baseline reference)
- **Step 2**: Window size reports + optimal window config
- **Step 3**: Master CSV files with 12 features
- **Step 4**: Final datasets with selected features + evaluation reports

---

### Complete Pipeline Command

Run all 4 steps sequentially:

```bash
java -cp ".:lib/weka.jar" ExperimentRunner experiment
```

This executes:
1. Format raw data → `formatted_data/`
2. Baseline evaluation → `results/1_baseline/`
3. Window optimization → `results/2_window_optimization/`
4. Feature expansion → `results/3_feature_expansion/`
5. Sequential feature selection → `results/4_feature_selection/`

**Total Execution Time**: ~10-30 minutes depending on data size and number of CSV files

**Automatic Logging**: Console output saved to `*_log.txt` in each step directory


## Results Directory Structure

### Numbered Steps (1→2→3→4)

**Step 1: `results/1_baseline/`** - Initial performance with 6 features, 1s window
- `baseline_log.txt` + 3 classifier directories (j48/, random_forest/, smo/)
- Each classifier: evaluation_report.txt, confusion_matrix.csv/png

**Step 2: `results/2_window_optimization/`** - Find optimal window size (test 1s-4s)
- `window_optimization_log.txt` + 3 classifier directories
- Each classifier: 4 window sizes × (report + CSV) + best PNG

**Step 3: `results/3_feature_expansion/`** - Create 12-feature master datasets
- `feature_expansion_log.txt` + 3 master CSVs (one per classifier with its optimal window)

**Step 4: `results/4_feature_selection/`** - SFS selects optimal features
- `feature_selection_log.txt` + 3 SFS subdirectories (sfs_j48/, sfs_random_forest/, sfs_smo/)
- Each: selected features CSV + evaluation_report.txt + confusion_matrix.csv/png

**Summary**: ~55 files (text/CSV), ~64 files (with PNG images), ~1-3 MB total

**Important Notes**:
- All `*_log.txt` files contain complete console output from that step
- Confusion matrix CSVs must be converted to PNG using `plot_confusion_matrix.py`
- Each classifier directory is independent (can analyze separately)
- SFS CSV format: **class column MUST be last** (Weka compatibility)

---

## Evaluation Reporting

The system generates comprehensive evaluation reports with detailed metrics for each classifier:

### Console Output
When running experiments, all evaluation metrics are printed to console:
- **Accuracy** - Overall classification accuracy (0-100%)
- **Kappa Statistic** - Agreement beyond chance (0-1, higher is better)
- **Confusion Matrix** - Complete matrix with row/column labels showing true vs predicted
- **Per-Class Metrics**:
  - **Precision** - Positive predictive value (what % of predicted X are actually X)
  - **Recall** - Sensitivity, true positive rate (what % of actual X are detected)
  - **F1-Score** - Harmonic mean of precision and recall (balanced metric)

### Automatic Logging

All console output is automatically saved to log files:
- `results/1_baseline/baseline_log.txt`
- `results/2_window_optimization/window_optimization_log.txt`
- `results/3_feature_expansion/feature_expansion_log.txt`
- `results/4_feature_selection/feature_selection_log.txt`

This creates a complete permanent record of all experiments without re-running.

### Saved Reports

For each classifier evaluation, the following files are automatically generated:

#### 1. **Detailed Text Report** (`<experiment_name>_evaluation_report.txt`)
- Timestamp and experiment information
- All metrics in formatted text:
  - Accuracy percentage
  - Kappa statistic
  - Complete confusion matrix with class labels
  - Per-class precision, recall, F1-score
- Full Weka evaluation summary

**Example Content**:
```
Evaluation Report: Baseline_RandomForest
Timestamp: 2025-12-04 15:30:45

Overall Accuracy: 97.20%
Kappa Statistic: 0.9580

Confusion Matrix:
                 bicep_curl  hammer_curl  tricep_kickback
bicep_curl              145            3                2
hammer_curl               2          148                0
tricep_kickback           1            0              149

Per-Class Metrics:
Class            Precision   Recall   F1-Score
bicep_curl          0.980    0.967     0.973
hammer_curl         0.980    0.987     0.983
tricep_kickback     0.987    0.993     0.990
```

#### 2. **Confusion Matrix CSV** (`<experiment_name>_confusion_matrix.csv`)
- Machine-readable format with class labels as headers
- Ready for visualization and analysis
- Can be processed by `plot_confusion_matrix.py`

**Format**:
```csv
,bicep_curl,hammer_curl,tricep_kickback
bicep_curl,145,3,2
hammer_curl,2,148,0
tricep_kickback,1,0,149
```

#### 3. **Confusion Matrix PNG** (`<experiment_name>_confusion_matrix.png`)
- Generated using Python visualization script
- High-quality heatmap image
- Color-coded for easy interpretation
- 300 DPI resolution (publication quality)

### Visualizing Confusion Matrices

A Python script is provided to generate heatmap images from CSV files:

**Setup** (one-time):
```bash
# Option 1: Using provided Conda environment
conda env create -f environment.yml
conda activate spml-viz

# Option 2: Using pip
pip install numpy matplotlib seaborn pandas
```

**Generate Images**:
```bash
# Single confusion matrix
python plot_confusion_matrix.py results/1_baseline/j48/Baseline_J48_confusion_matrix.csv

# All confusion matrices in a directory
python plot_confusion_matrix.py results/1_baseline/

# All confusion matrices in entire results tree
python plot_confusion_matrix.py results/
```

**Output Features**:
- Color-coded heatmap (blue gradient - darker = more predictions)
- Annotated cell values (actual counts)
- Labeled axes with class names
- Title with experiment name
- 300 DPI resolution for presentations/papers

**Example Output**: `results/1_baseline/j48/Baseline_J48_confusion_matrix.png`

### Report Organization

All reports follow a consistent naming pattern:
```
<ExperimentType>_<Classifier>_<Details>_<FileType>

Examples:
Baseline_J48_evaluation_report.txt
WindowOpt_RandomForest_2.0s_confusion_matrix.csv
SFS_SMO_6features_evaluation_report.txt
```

This makes it easy to:
- Identify which experiment generated the report
- Compare same metric across classifiers
- Track optimization progress
- Share specific results with collaborators



## Usage

### Important: Version Control & Data Management

**What's in Git (version controlled):**
- ✅ All Java source files (*.java)
- ✅ Configuration files (config.json, environment.yml)
- ✅ Scripts (compile.sh, check_jdk.sh, plot_confusion_matrix.py)
- ✅ Documentation (README.md, .gitignore)
- ✅ Project structure and architecture

**What's NOT in Git (gitignored - generated/downloaded locally):**
- ❌ `raw_data/` - Your accelerometer CSV files (dataset-specific, too large)
- ❌ `formatted_data/` - Generated from raw_data processing
- ❌ `results/` - All experiment outputs (logs, reports, CSVs, images)
- ❌ `lib/weka.jar` - Downloaded separately (large binary, 70+ MB)
- ❌ `*.class` - Compiled Java bytecode
- ❌ `*.arff` - Generated Weka format files
- ❌ `data*/` folders - Any directory starting with "data"

**Why this separation:**
- **Clean repository** - Only source code and documentation in Git
- **No bloat** - Data and results can be gigabytes, Git repos stay small
- **Privacy** - Each user's data stays local
- **Reproducibility** - Anyone can clone and generate their own results
- **Flexibility** - Different users can work with different datasets

**Workflow Implication**:
1. Clone repository → Get only code
2. Download Weka → One-time setup
3. Add your data → Local only
4. Run experiments → Results stay local
5. Share code changes → Push to Git
6. Share results separately → Email, cloud storage, etc.

### Prerequisites

#### 1. Java JDK

Check installation: `./check_jdk.sh`

Install if needed:
```bash
# Ubuntu/Debian
sudo apt install default-jdk

# macOS
brew install openjdk
```

**Required**: Java 8+, **Tested with**: OpenJDK 21

---

#### 2. Weka Library

Download and extract to `lib/`:
```bash
cd lib
wget https://sourceforge.net/projects/weka/files/weka-3-8/3.8.6/weka-3-8-6.zip
unzip weka-3-8-6.zip && mv weka-3-8-6/weka.jar . && rm -rf weka-3-8-6*
cd ..
```

Or download manually from: https://sourceforge.net/projects/weka/files/weka-3-8/3.8.6/

---

#### 3. Get Raw Data

**Download from Google Drive**: [Link will be provided - raw accelerometer CSV files]

**Setup**:
```bash
# Download and extract to raw_data/
unzip raw_data.zip -d raw_data/

# Verify files
ls raw_data/*.csv
```

**File naming format**: `WatchID-Project-Subject-Placement-Activity-Info-DateTime.csv`
- Activity field determines class label (e.g., `bicep_curl`, `hammer_curl`, `tricep_kickback`)

---

#### 4. Python (Optional - for visualization)

```bash
# Option 1: Conda environment
conda env create -f environment.yml
conda activate spml-viz

# Option 2: pip
pip install numpy matplotlib seaborn pandas
```

Only needed for generating confusion matrix PNG images.

---

### Compile

```bash
./compile.sh
```

1. Removes all `*.class` files (clean build)
2. Compiles all Java files with Weka in classpath
3. Shows compilation progress
4. Reports success/failure

**Manual compilation** (if script doesn't work):
```bash
# Linux/macOS
javac -cp ".:lib/weka.jar" *.java

# Windows
javac -cp ".;lib/weka.jar" *.java
```

**Clean only** (remove .class files without recompiling):
```bash
./compile.sh --clean
```

**Troubleshooting**:
- If "command not found": Make script executable with `chmod +x compile.sh`
- If compilation fails: Check that `lib/weka.jar` exists

Compiles all Java files with Weka in classpath. Use `./compile.sh --clean` to remove .class files only.

---

### Run Experiments

#### Complete Pipeline (Recommended)

```bash
java -cp ".:lib/weka.jar" ExperimentRunner experiment
```

Runs all 4 steps: baseline → window optimization → feature expansion → feature selection

**Execution time**: ~10-30 minutes  
**Outputs**: Console + log files in each results subdirectory

---

#### Individual Commands

```bash
# Step 1: Baseline (6 features, 1s window)
java -cp ".:lib/weka.jar" ExperimentRunner baseline

# Step 2: Window optimization (test 1s-4s)
java -cp ".:lib/weka.jar" ExperimentRunner optimize

# Step 3: Feature expansion (create 12-feature datasets)
java -cp ".:lib/weka.jar" ExperimentRunner features

# Step 4: SFS feature selection
java -cp ".:lib/weka.jar" ExperimentRunner selection

# Format raw data only
java -cp ".:lib/weka.jar" ExperimentRunner format

# Help
java -cp ".:lib/weka.jar" ExperimentRunner help
```

---

#### Custom Experiments

```bash
java -cp ".:lib/weka.jar" ExperimentRunner custom <features> <window> <classifier> [sfs]
```

- `features`: `basic` (6) or `expanded` (12)
- `window`: 1000-4000 ms
- `classifier`: `J48`, `RF`, or `SVM`
- `sfs`: Optional feature selection flag

**Examples**:
```bash
java -cp ".:lib/weka.jar" ExperimentRunner custom basic 2000 J48
java -cp ".:lib/weka.jar" ExperimentRunner custom expanded 3000 RF sfs
```

---

#### Help and Usage

**Show all available commands**:
```bash
java -cp ".:lib/weka.jar" ExperimentRunner help
```

**Available commands**:
- `experiment` - Complete 4-step pipeline
- `baseline` - Step 1: Baseline evaluation
- `optimize` - Step 2: Window optimization
- `features` - Step 3: Feature expansion
- `selection` / `compare` - Step 4: Sequential feature selection
- `format` - Format raw data only
- `extract` - Extract basic features only
- `custom <features> <window> <classifier> [sfs]` - Custom experiment

---

#### Generate Confusion Matrix Images

After running experiments, visualize confusion matrices:

**Single file**:
```bash
python plot_confusion_matrix.py results/1_baseline/j48/Baseline_J48_confusion_matrix.csv
```

**Entire directory** (all CSVs):
```bash
python plot_confusion_matrix.py results/1_baseline/
```

**All results** (recursively):
```bash
python plot_confusion_matrix.py results/
```

---

#### Generate Confusion Matrix Images

```bash
# Single file
python plot_confusion_matrix.py results/1_baseline/j48/Baseline_J48_confusion_matrix.csv

# Entire directory
python plot_confusion_matrix.py results/

```

Generates 300 DPI PNG heatmaps next to each confusion matrix CSV.

---

### Quick Start Workflow

```bash
# 1. Setup (once)
git clone https://github.com/jinyoonok2/SPML-course-projects.git
cd SPML-course-projects
# Download Weka to lib/ and raw data to raw_data/

# 2. Compile
./compile.sh

# 3. Run experiments
java -cp ".:lib/weka.jar" ExperimentRunner experiment

# 4. Visualize (optional)
python plot_confusion_matrix.py results/

# 5. Check results
cat results/4_feature_selection/feature_selection_log.txt
```

---

## Technical Details

### Features

**Basic (6)**: mean, std per axis (X, Y, Z)  
**Expanded (12)**: Basic + median, RMS per axis

Extracted using sliding windows (1-4 seconds, 1s stride) from 50Hz accelerometer data.
**When to use**: Feature expansion step, SFS (provides more options for selection), final optimized models

---

### Classifiers

**J48** (Decision Tree): Fast, interpretable, good for understanding feature importance  
**Random Forest**: Ensemble method, highest accuracy, robust to overfitting  
**SVM** (SMO): Good for high-dimensional data, margin-based classification

Each classifier independently optimized with its own window size and feature subset.

---

### Sequential Feature Selection (SFS)

**Forward selection** starting from empty set:
1. Try adding each feature individually
2. Keep feature with best accuracy improvement
3. Repeat until improvement < 0.001 (0.1%)
4. Return selected features

Prevents overfitting by eliminating redundant features. Each classifier selects different optimal subset (typically 5-10 features from 12).

---

### Evaluation

**10-fold cross-validation** for all experiments:
- Split data into 10 parts
- Train on 9, test on 1
- Repeat 10 times, average results

**Metrics**:
- **Accuracy**: Overall correct predictions (%)
- **Kappa**: Agreement beyond chance (0-1)
- **Confusion Matrix**: Actual vs predicted counts
- **Per-Class**: Precision, Recall, F1-Score

All metrics saved to evaluation reports in results directories.

---

## Dependencies

**Java**: JDK 8+ (tested with OpenJDK 21)  
**Weka**: 3.8.6 (download to `lib/weka.jar`)  
**Python** (optional): 3.8+ for visualization (matplotlib, seaborn, numpy, pandas)

---

## Configuration

**config.json** controls project settings:
```json
{
  "project": "FinalProject",
  "activities": ["bicep_curl", "hammer_curl", "tricep_kickback"],
  "subjects": ["Jinyoon", "Josh", "Katz"],
  "placements": ["left"]
}
```

Activities are auto-detected from filenames. Update `activities` array to match your dataset.

---

## Important Notes

- **File naming**: `WatchID-Project-Subject-Placement-Activity-Info-DateTime.csv`
- **Activity extraction**: Activity field from filename determines class label
- **SFS CSVs**: Class column MUST be last (Weka requirement)
- **Multi-class**: System auto-detects unique activities, no code changes needed
- **Results**: Gitignored - each user generates locally
- **Raw data**: Available via Google Drive (see Prerequisites section)

---

## Legacy Files

`legacy/` contains old Part1-5 assignment files (archived, not used):
- Replaced by new 7-component modular architecture
- Kept for reference only

---

## Troubleshooting

**"Class not found"**: Check `lib/weka.jar` exists, verify classpath separator (`:` vs `;`)  
**"No files in raw_data"**: Verify CSV filenames match expected pattern  
**Low accuracy**: Ensure sufficient data (>30s per activity), check activity labels  
**SFS parse error**: Verify class column is LAST in CSV

- Maintain consistent form across repetitions
- Include multiple subjects for generalization
- Balance number of samples across activities

**Experiments**:
- Always run baseline first (establishes reference)
- Let complete pipeline run uninterrupted (~10-30 min)
- Check log files if errors occur
- Generate PNG images after experiments for analysis

**Results Analysis**:
- Compare baseline vs final SFS accuracy
- Check confusion matrix to identify misclassifications
- Review which features were selected (in SFS CSVs)
- Look for consistent patterns across classifiers

**Code Modifications**:
- Edit `config.json` for new activities (no code changes needed)
- Modify `MIN_IMPROVEMENT` in OptimizationEngine.java for different SFS sensitivity
- Adjust window sizes in OptimizationEngine.java if needed
- Keep `compile.sh` executable: `chmod +x compile.sh`

**Sharing Results**:
- Zip `results/` directory for sharing
- Include log files for complete record
- Share PNG images for visual analysis
- Document any custom parameter changes

---

## Troubleshooting

**"Class not found" error**:
- Ensure `lib/weka.jar` exists and is ~70-80 MB
- Check classpath separator (`:` for Linux/Mac, `;` for Windows)
- Verify compilation: `ls *.class` should show many files

**"No files found in raw_data"**:
- Check directory exists: `ls -la raw_data/`
- Verify CSV filenames match expected pattern
- Check file permissions (must be readable)

**SFS CSV parse error**:
- Ensure class column is LAST in SFS-generated CSVs
- Check for missing commas or malformed CSV
- Verify class values match @ATTRIBUTE definition

**Low accuracy (<80%)**:

---

## Credits

**Author**: Jinyoon Ok  
**Repository**: https://github.com/jinyoonok2/SPML-course-projects  
**Version**: 2.0 (Modular Architecture with Automatic Logging)  
**Last Updated**: December 2025


