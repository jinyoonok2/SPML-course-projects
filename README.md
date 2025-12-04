# SPML Course - Activity Recognition System

## Overview
A modular machine learning system for activity recognition using smartwatch accelerometer data. The project supports multi-class classification for any activities with automatic class detection, per-classifier optimization, and comprehensive feature engineering.

**Current Application**: Dumbbell Exercise Classification
- **Class 1**: Bicep Curl (Supinated grip)
- **Class 2**: Hammer Curl (Neutral grip)  
- **Class 3**: Tricep Kickback (Linear arm extension)

Using accelerometer data from left wrist to distinguish between easily confused dumbbell exercises.

**Key Features**: 
- **Multi-class support** - Automatically detects and classifies any number of activities
- **Per-classifier optimization** - Each classifier gets its own optimal window size and feature set
- **Modular architecture** - 5 specialized engines for clean separation of concerns
- **Flexible data format** - Accepts any project identifier in filenames

## Project Structure

```
SPML-course-projects/
├── raw_data/                           # All raw CSV files from smartwatch (gitignored)
│   └── WatchID-FinalProject-Subject-left-Activity-Info-DateTime.csv
│       # Activity: bicep_curl, hammer_curl, tricep_kickback
│       # Examples: Watch123-FinalProject-Jinyoon-left-bicep_curl-Info-Date.csv
├── formatted_data/                     # Cleaned data (timestamp, ax, ay, az) (gitignored)
├── results/                            # Experiment results (gitignored - generated locally)
│   ├── 1_baseline/                     # Step 1: Baseline evaluation (6 features, 1s window)
│   │   ├── j48/
│   │   │   ├── Baseline_J48_evaluation.txt
│   │   │   ├── Baseline_J48_confusion_matrix.csv
│   │   │   └── Baseline_J48_confusion_matrix.png
│   │   ├── random_forest/
│   │   └── smo/
│   ├── 2_window_optimization/          # Step 2: Window size tuning (test 1s-4s)
│   │   ├── j48/
│   │   ├── random_forest/
│   │   └── smo/
│   ├── 3_feature_expansion/            # Step 3: Expand to 12 features (master datasets)
│   │   ├── j48_master_dataset_12features.csv
│   │   ├── random_forest_master_dataset_12features.csv
│   │   └── smo_master_dataset_12features.csv
│   └── 4_feature_selection/            # Step 4: SFS - Select best features
│       ├── sfs_j48/
│       │   ├── SFS_J48_7features_evaluation.txt
│       │   └── SFS_J48_7features_confusion_matrix.csv
│       ├── sfs_random_forest/
│       └── sfs_smo/
├── legacy/                             # Old Part1-5 files (archived)
│   ├── Part1_DataProcessor.java
│   ├── Part2_WindowTuning.java
│   ├── Part3_FeatureExpansion.java
│   ├── Part4_FeatureSelection.java
│   ├── Part5_ClassifierComparison.java
│   ├── WadaManager_old.java
│   └── Assignment2_Handler.java
├── lib/
│   └── weka.jar                        # Weka library 3.8.6 (gitignored - download separately)
├── check_jdk.sh                        # Script to verify JDK installation
├── compile.sh                          # Script to clean and compile all Java files
├── plot_confusion_matrix.py            # Python script to generate confusion matrix images
├── config.json                         # Project configuration (activities, placements)
├── ExperimentRunner.java               # Main entry point
├── DataManager.java                    # Data processing engine
├── FeatureEngine.java                  # Feature extraction engine
├── OptimizationEngine.java             # Optimization and SFS engine
├── MLEngine.java                       # ML operations engine
├── EvaluationReporter.java             # Comprehensive evaluation reporting
├── MyWekaUtils.java                    # Core Weka utilities
└── README.md
```

**Note:** Files marked as "gitignored" are generated locally or too large for version control. See `.gitignore` for complete list.

### New Modular Architecture (5 Engines)

**ExperimentRunner** - Main entry point and command interface
- Orchestrates all experiments
- Command-line interface for running experiments
- Supports both structured and custom modular experiments

**DataManager** - Raw data processing and format conversion
- Format raw CSV files (drop sensor_type, accuracy columns)
- Convert CSV to ARFF format
- Extract activity labels from filenames (supports any activity type)
- Automatic multi-class detection from file naming

**FeatureEngine** - Feature extraction and engineering
- Extract basic features (6): mean, std per axis
- Extract expanded features (12): mean, std, median, RMS per axis
- Sliding window feature extraction (configurable window size)

**OptimizationEngine** - Parameter optimization and feature selection
- Window size optimization per classifier (tests 1s-4s windows)
- Sequential Feature Selection (SFS) per classifier
- Saves/loads optimization configurations

**MLEngine** - Machine learning operations and evaluation
- Modular experiment configuration system
- Per-classifier baseline evaluation
- Complete experiment pipeline orchestration
- Result reporting and persistence

**MyWekaUtils** - Core Weka utilities (unchanged from original)

## Experiment Workflow

### Per-Classifier Optimization Pipeline

The new architecture implements a per-classifier optimization workflow where each classifier (J48, RandomForest, SVM) gets its own:

1. **Baseline Evaluation** (Step 1)
   - All classifiers tested with 6 basic features, 1s window
   - Establishes baseline performance for each classifier

2. **Window Optimization** (Step 2)
   - Each classifier tested with 1s, 2s, 3s, 4s windows
   - Each classifier gets its own optimal window size
   - Results saved per classifier

3. **Master Dataset Generation** (Step 3)
   - Each classifier gets a master dataset with:
     - 12 expanded features (mean, std, median, RMS per axis)
     - Its own optimal window size from Step 2
   - 3 separate master datasets created (one per classifier)

4. **Sequential Feature Selection** (Step 4)
   - SFS performed on each classifier's master dataset
   - Each classifier selects its own optimal feature subset
   - Forward selection with MIN_IMPROVEMENT threshold (0.001)

5. **Final Comparison** (Step 5)
   - Compare all classifiers using their SFS-selected features
   - SFS results ARE the final comparison (no separate evaluation)
   - Determine best overall classifier and configuration

### Modular Experiment Configuration

The new `ExperimentConfig` system allows mixing and matching any combination:
- **Features**: Basic (6) or Expanded (12)
- **Window**: Any size (1000ms - 4000ms)
- **Classifier**: J48 (Decision Tree), RandomForest, or SVM
- **Feature Selection**: On or Off

This enables flexible experimentation and custom workflows.

## Evaluation Reporting

The system generates comprehensive evaluation reports with detailed metrics for each classifier:

### Console Output
When running experiments, all evaluation metrics are printed to console:
- **Accuracy** - Overall classification accuracy
- **Kappa Statistic** - Agreement beyond chance
- **Confusion Matrix** - Complete matrix with row/column labels
- **Per-Class Metrics**:
  - Precision (positive predictive value)
  - Recall (sensitivity)
  - F1-Score (harmonic mean of precision and recall)

### Saved Reports

For each classifier evaluation, the following files are automatically generated:

1. **Detailed Text Report** (`<experiment_name>_evaluation_report.txt`)
   - Timestamp and experiment information
   - All metrics in formatted text
   - Complete confusion matrix with labels
   - Full Weka evaluation summary

2. **Confusion Matrix CSV** (`<experiment_name>_confusion_matrix.csv`)
   - Machine-readable format with class labels
   - Ready for visualization and analysis
   - Can be converted to images using provided Python script

### Visualizing Confusion Matrices

A Python script is provided to generate heatmap images from CSV files:

```bash
# Install required Python packages
pip install numpy matplotlib seaborn

# Generate image for single confusion matrix
python plot_confusion_matrix.py results/baseline/j48/Baseline_J48_confusion_matrix.csv

# Generate images for all confusion matrices in results directory
python plot_confusion_matrix.py results/
```

Output: High-quality PNG images with:
- Color-coded heatmap (blue gradient)
- Annotated cell values
- Labeled axes with class names
- 300 DPI resolution for publication

### Report Organization

Reports are organized by experiment type and classifier:
```
results/
├── baseline/
│   ├── j48/
│   │   ├── Baseline_J48_evaluation_report.txt
│   │   ├── Baseline_J48_confusion_matrix.csv
│   │   └── Baseline_J48_confusion_matrix.png
│   ├── random_forest/
│   └── smo/
├── window_optimization/
│   ├── j48/
│   │   ├── WindowOpt_J48_2.0s_evaluation_report.txt
│   │   └── WindowOpt_J48_2.0s_confusion_matrix.csv
│   └── ...
└── experiment/
    ├── j48/
    │   └── sfs_j48/
    │       ├── SFS_J48_7features_evaluation_report.txt
    │       └── SFS_J48_7features_confusion_matrix.csv
    └── ...
```

## Usage

### Important: Version Control & Data Management

**What's in Git (version controlled):**
- ✅ All Java source files (.java)
- ✅ Configuration files (config.json)
- ✅ Scripts (compile.sh, check_jdk.sh, plot_confusion_matrix.py)
- ✅ Documentation (README.md)
- ✅ Project structure (.gitignore)

**What's NOT in Git (gitignored - generated/downloaded locally):**
- ❌ `raw_data/` - Your accelerometer CSV files (too large, dataset-specific)
- ❌ `formatted_data/` - Generated from raw_data
- ❌ `results/` - All experiment outputs (generated locally)
- ❌ `lib/weka.jar` - Download separately (large file)
- ❌ Compiled files (*.class, *.arff)

**Why this matters:**
- Data and results are NOT committed to avoid repository bloat
- Each user maintains their own data and results locally
- Weka library must be downloaded once per setup
- This keeps the repository clean and focused on code

### Prerequisites

1. **Install JDK** (if not already installed):
```bash
# Ubuntu/Debian
sudo apt install default-jdk

# Check installation
./check_jdk.sh
```

2. **Download Weka Library**:
```bash
# Download Weka 3.8.6
cd lib
wget https://sourceforge.net/projects/weka/files/weka-3-8/3.8.6/weka-3-8-6.zip
unzip weka-3-8-6.zip
mv weka-3-8-6/weka.jar .
rm -rf weka-3-8-6 weka-3-8-6.zip
cd ..
```

3. **Prepare Data Directory**:
```bash
# Create raw_data directory for your CSV files
mkdir -p raw_data

# Place your smartwatch data files here
# Format: WatchID-FinalProject-Subject-left-ActivityName-Info-DateTime.csv
```
```

### Compile

```bash
./compile.sh
```

This script:
- Removes all existing .class files
- Compiles all Java files
- Verifies successful compilation

### Run Experiments

#### Complete Pipeline (Recommended for first run)
```bash
# Run complete per-classifier optimization pipeline
java -cp ".:lib/weka.jar" ExperimentRunner experiment
```

This executes:
1. Baseline evaluation (all classifiers)
2. Window optimization (per classifier)
3. Master dataset generation (per classifier)
4. Sequential feature selection (per classifier)
5. Final comparison report

#### Individual Commands

```bash
# 1. Baseline evaluation (all classifiers, 6 features, 1s window)
java -cp ".:lib/weka.jar" ExperimentRunner baseline

# 2. Window optimization (find optimal window per classifier)
java -cp ".:lib/weka.jar" ExperimentRunner optimize

# 3. Feature expansion (create master datasets per classifier)
java -cp ".:lib/weka.jar" ExperimentRunner features

# 4. Sequential feature selection (per classifier)
java -cp ".:lib/weka.jar" ExperimentRunner selection
# OR
java -cp ".:lib/weka.jar" ExperimentRunner compare

# Data processing only
java -cp ".:lib/weka.jar" ExperimentRunner format   # Format raw data
java -cp ".:lib/weka.jar" ExperimentRunner extract  # Extract basic features
```

#### Modular Custom Experiments

Mix and match any configuration:

```bash
# Custom experiment: <features> <window> <classifier> [sfs]
# Features: 'basic' (6) or 'expanded' (12)
# Window: 1000-4000 (milliseconds)
# Classifier: 'J48', 'RF', or 'SVM'
# SFS: optional flag to enable feature selection

# Example 1: Basic features, 2s window, J48
java -cp ".:lib/weka.jar" ExperimentRunner custom basic 2000 J48

# Example 2: Expanded features, 3s window, SVM, with SFS
java -cp ".:lib/weka.jar" ExperimentRunner custom expanded 3000 SVM sfs

# Example 3: Expanded features, 1s window, RandomForest
java -cp ".:lib/weka.jar" ExperimentRunner custom expanded 1000 RF

# Example 4: Basic features, 4s window, SVM, with SFS
java -cp ".:lib/weka.jar" ExperimentRunner custom basic 4000 SVM sfs
```

#### Help

```bash
# Show all available commands
java -cp ".:lib/weka.jar" ExperimentRunner help
```

## Key Features

### Feature Extraction
- **Basic Features (6)**:
  - mean_x, std_x
  - mean_y, std_y
  - mean_z, std_z

- **Expanded Features (12)**:
  - mean_x, std_x, median_x, rms_x
  - mean_y, std_y, median_y, rms_y
  - mean_z, std_z, median_z, rms_z

### Time Windows
- **Fixed Windows**: 1, 2, 3, 4 seconds
- **Sliding Window**: 1-second stride for multi-second windows

### Classifiers
- **Decision Tree** (J48 in Weka)
- **Random Forest**
- **SVM** (SMO in Weka)

## Data Format

### Raw Data CSV
```
timestamp_ms, sensor_type, accuracy, ax, ay, az
1759458897308,1,3,-5.1217494,-0.86579955,8.483159
```

### Features CSV
```
mean_x,std_x,mean_y,std_y,mean_z,std_z,Activity
-4.523,0.421,1.234,0.567,8.901,0.345,bicep_curl
-2.134,0.823,0.456,0.234,9.234,0.567,hammer_curl
-1.234,0.456,0.789,0.123,8.567,0.234,tricep_kickback
```

### ARFF Format (Weka)
```
@RELATION dumbbell_exercises

@ATTRIBUTE mean_x NUMERIC
@ATTRIBUTE std_x NUMERIC
@ATTRIBUTE mean_y NUMERIC
@ATTRIBUTE std_y NUMERIC
@ATTRIBUTE mean_z NUMERIC
@ATTRIBUTE std_z NUMERIC
@ATTRIBUTE Activity {bicep_curl,hammer_curl,tricep_kickback}

@DATA
-4.523,0.421,1.234,0.567,8.901,0.345,bicep_curl
-2.134,0.823,0.456,0.234,9.234,0.567,hammer_curl
```

## Platform Compatibility

This project is compatible with **Linux, macOS, and Windows** (with minor adjustments):

### Linux & macOS ✅
- Shell scripts (`check_jdk.sh`, `compile.sh`) work natively
- All commands work as-is

### Windows
For Windows users, you have two options:

**Option 1: Use WSL (Windows Subsystem for Linux)** - Recommended
- Install WSL and run commands as-is
- Everything works the same as Linux

**Option 2: Use Command Prompt/PowerShell**
- Replace shell scripts with manual commands:
  ```bash
  # Instead of ./compile.sh, use:
  javac -cp ".;lib/weka.jar;lib/ascii-table-1.2.0.jar" *.java
  
  # Instead of running programs with:
  java -cp ".:lib/weka.jar:lib/ascii-table-1.2.0.jar" Assignment2_Handler part1
  # Use (note semicolons instead of colons):
  java -cp ".;lib/weka.jar;lib/ascii-table-1.2.0.jar" Assignment2_Handler part1
  ```
- **Note**: Windows uses `;` as classpath separator, not `:`

## Dependencies
- **Java**: JDK 8 or higher (tested with OpenJDK 21)
- **Weka**: 3.8.6 (place `weka.jar` in `lib/` directory)
  - Download: https://sourceforge.net/projects/weka/files/weka-3-8/3.8.6/
  - Or use wget (see Prerequisites section above)

## Output Files

Results are organized by experiment type:

```
results/
├── baseline/
│   └── baseline_results.txt              # All classifiers baseline
├── window_optimization/
│   ├── j48_window_optimization.txt       # J48 window results (1s-4s)
│   ├── random_forest_window_optimization.txt
│   ├── svm_window_optimization.txt
│   └── optimal_window_config.txt         # Saved optimal windows
├── master_datasets/
│   ├── j48_master_dataset_12features.csv          # J48's optimal config
│   ├── random_forest_master_dataset_12features.csv
│   └── svm_master_dataset_12features.csv
├── feature_selection/
│   ├── j48_sfs_results.txt               # J48 SFS progression
│   ├── random_forest_sfs_results.txt
│   └── svm_sfs_results.txt
└── final_report/
    └── experiment_report.txt             # Complete summary
```

## Legacy Files

Old Part1-5 files have been moved to `legacy/` folder for reference:
- `Part1_DataProcessor.java` → Replaced by DataManager + MLEngine baseline
- `Part2_WindowTuning.java` → Replaced by OptimizationEngine window optimization
- `Part3_FeatureExpansion.java` → Replaced by FeatureEngine expanded features
- `Part4_FeatureSelection.java` → Replaced by OptimizationEngine SFS
- `Part5_ClassifierComparison.java` → Replaced by MLEngine complete pipeline
- `WadaManager_old.java` → Replaced by ExperimentRunner
- `Assignment2_Handler.java` → Replaced by ExperimentRunner

These files are kept for reference but are no longer used in the new architecture.

## Notes
- **Current Project**: Dumbbell Exercise Classification (3 classes)
  - `bicep_curl` - Supinated grip bicep curl
  - `hammer_curl` - Neutral grip hammer curl
  - `tricep_kickback` - Linear arm extension
- **Filename Format**: `WatchID-FinalProject-Subject-left-Activity-Info-DateTime.csv`
  - Project field: `FinalProject` (or any identifier)
  - Placement: `left` (left wrist)
  - Activity field: Determines the class label (bicep_curl, hammer_curl, tricep_kickback)
- **Multi-class Support**: Automatically detects all unique activities from filenames
- **Sensor**: Accelerometer data at 50Hz sampling rate
- Classification uses 10-fold cross-validation for accuracy evaluation
- Sequential Feature Selection uses forward selection with MIN_IMPROVEMENT = 0.001
- Each classifier is optimized independently for maximum performance
- SFS results are the final comparison (no separate evaluation step needed)

## Platform Compatibility

This project is compatible with **Linux, macOS, and Windows**:

### Linux & macOS ✅
- Shell scripts (`check_jdk.sh`, `compile.sh`) work natively
- Use `:` as classpath separator
- All commands work as shown above

### Windows
**Option 1: WSL (Windows Subsystem for Linux)** - Recommended
- Install WSL and run commands as-is
- Everything works the same as Linux

**Option 2: Command Prompt/PowerShell**
- Replace `:` with `;` in classpath:
  ```bash
  # Instead of:
  java -cp ".:lib/weka.jar" ExperimentRunner experiment
  
  # Use:
  java -cp ".;lib/weka.jar" ExperimentRunner experiment
  ```
- Manually compile:
  ```bash
  javac -cp ".;lib/weka.jar" *.java
  ```

## Date
December 2025
