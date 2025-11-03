# SPML Course - Assignment 2: Gesture Recognition Optimization

## Overview
This assignment builds upon Assignment 1 to improve gesture recognition accuracy through parameter tuning, feature engineering, and classifier comparison. The project uses smartwatch accelerometer data to classify hand-washing gestures.

## Project Structure

```
SPML-course-projects/
├── raw_data/                           # All raw CSV files from smartwatch (Assignment 1 + 2)
│   └── G5NZCJ022402200-Assignment*-Jinyoon-*-hand_wash-*.csv
├── formatted_data/                     # Cleaned data (timestamp, ax, ay, az)
├── results/                            # Results organized by part
│   ├── part1/                          # Part 1 results
│   │   ├── features.csv
│   │   ├── features.arff
│   │   └── results.txt
│   ├── part2/                          # Part 2 results (different window sizes)
│   │   ├── features_1s.csv
│   │   ├── features_1s.arff
│   │   ├── features_2s.csv
│   │   ├── features_2s.arff
│   │   ├── features_3s.csv
│   │   ├── features_3s.arff
│   │   ├── features_4s.csv
│   │   ├── features_4s.arff
│   │   ├── best_window_config.txt
│   │   └── results.txt
│   ├── part3/                          # Part 3 results (12 features)
│   │   ├── features.csv
│   │   ├── features.arff
│   │   └── results.txt
│   ├── part4/                          # Part 4 results (feature selection)
│   └── part5/                          # Part 5 results (classifier comparison)
├── lib/
│   ├── weka.jar                        # Weka library (download separately)
│   └── ascii-table-1.2.0.jar           # ASCII Table library (optional)
├── check_jdk.sh                        # Script to verify JDK and Weka installation
├── compile.sh                          # Script to clean and compile all Java files
├── Assignment2_Handler.java            # Main handler/entry point for all parts
├── Part1_DataProcessor.java            # Part 1: Process all data & baseline Decision Tree
├── Part2_WindowTuning.java             # Part 2: Time window experiments (1s, 2s, 3s, 4s)
├── Part3_FeatureExpansion.java         # Part 3: Add median & RMS features
├── Part4_FeatureSelection.java         # Part 4: Sequential feature selection (Decision Tree)
├── Part5_ClassifierComparison.java     # Part 5: Compare Random Forest & SVM
└── README.md
```

Each Part file contains all necessary functions for feature extraction, ARFF conversion, and Weka classification.

## Assignment Parts

### Part 1: Process All Data with Decision Tree Baseline
- **Goal**: Process all raw data (Assignment 1 + 2 combined) and establish baseline accuracy
- **Implementation**: `Part1_DataProcessor.java`
- **Features**: 6 features (mean, std per axis)
- **Window**: 1 second
- **Classifier**: Decision Tree
- **Output**: Baseline accuracy with complete dataset

### Part 2: Time Window Tuning
- **Goal**: Test different time windows (1s, 2s, 3s, 4s) with 1-second sliding window
- **Implementation**: `Part2_WindowTuning.java`
- **Features**: 6 features (mean, std per axis)
- **Classifier**: Decision Tree
- **Output**: Accuracy for each window size to identify optimal time interval
- **Note**: Saves best window size to `best_window_config.txt` for use in Parts 3, 4, 5

### Part 3: Feature Expansion
- **Goal**: Add median and RMS (Root Mean Square) features
- **Implementation**: `Part3_FeatureExpansion.java`
- **Features**: 12 features (mean, std, median, RMS per axis)
- **Window**: Best window from Part 2 (automatically loaded from config)
- **Classifier**: Decision Tree
- **Output**: Accuracy improvement with expanded features

### Part 4: Sequential Feature Selection (Decision Tree)
- **Goal**: Find optimal feature subset using forward selection
- **Implementation**: `Part4_FeatureSelection.java`
- **Features**: Sequential selection from 12 features
- **Window**: Best window from Part 2
- **Classifier**: Decision Tree
- **Output**: 
  - Selected feature subset
  - Accuracy progression as features are added
  - Comparison with Part 3 (all 12 features)

### Part 5: Classifier Comparison
- **Goal**: Compare Random Forest and SVM with feature selection
- **Implementation**: `Part5_ClassifierComparison.java`
- **Features**: Sequential selection from 12 features (separate for each classifier)
- **Window**: Best window from Part 2
- **Classifiers**: Random Forest, SVM
- **Output**: 
  - Selected features for each classifier
  - Accuracy progression for each classifier
  - Best classifier overall (Decision Tree vs. Random Forest vs. SVM)

## Usage

### Check System Requirements
```bash
./check_jdk.sh
```
This script verifies:
- JDK installation and version (requires JDK 8+)
- Weka library presence (lib/weka.jar)

### Compile All Java Files
```bash
# Clean and compile
./compile.sh

# Clean only (remove .class files without recompiling)
./compile.sh --clean
```
This script:
- Removes all existing .class files
- Recompiles all Java files in the directory (unless using `--clean`)
- Verifies successful compilation

### Run Individual Parts

**Parts 1, 2, and 3:**
```bash
# Part 1: Data combination
java -cp ".:lib/weka.jar:lib/ascii-table-1.2.0.jar" Assignment2_Handler part1

# Part 2: Window tuning
java -cp ".:lib/weka.jar:lib/ascii-table-1.2.0.jar" Assignment2_Handler part2

# Part 3: Feature expansion
java -cp ".:lib/weka.jar:lib/ascii-table-1.2.0.jar" Assignment2_Handler part3
```

**Parts 4 and 5 (Recommended - avoids Java reflection warnings):**

Parts 4 and 5 use advanced Weka features that may trigger Java reflection warnings on Java 9+. To avoid these warnings, use the `--add-opens` flags:

```bash
# Part 4: Feature selection (Decision Tree)
java --add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.lang.reflect=ALL-UNNAMED -cp ".:lib/weka.jar:lib/ascii-table-1.2.0.jar" Assignment2_Handler part4

# Part 5: Classifier comparison (Random Forest & SVM)
java --add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.lang.reflect=ALL-UNNAMED -cp ".:lib/weka.jar:lib/ascii-table-1.2.0.jar" Assignment2_Handler part5
```

**Note**: The basic command without `--add-opens` flags will still work, but you may see Java reflection warnings in the console (these do not affect execution or results).

### Run All Parts
```bash
# Recommended (avoids Java reflection warnings for Parts 4 & 5)
java --add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.lang.reflect=ALL-UNNAMED -cp ".:lib/weka.jar:lib/ascii-table-1.2.0.jar" Assignment2_Handler all
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
-4.523,0.421,1.234,0.567,8.901,0.345,hand_wash
```

### ARFF Format (Weka)
```
@RELATION gestures

@ATTRIBUTE mean_x NUMERIC
@ATTRIBUTE std_x NUMERIC
@ATTRIBUTE mean_y NUMERIC
@ATTRIBUTE std_y NUMERIC
@ATTRIBUTE mean_z NUMERIC
@ATTRIBUTE std_z NUMERIC
@ATTRIBUTE Activity {hand_wash,non_hand_wash}

@DATA
-4.523,0.421,1.234,0.567,8.901,0.345,hand_wash
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
- **Java**: JDK 8 or higher
- **Weka**: 3.8.x or higher (place `weka.jar` in `lib/` directory)
  - Download from: https://www.cs.waikato.ac.nz/ml/weka/
- **ASCII Table**: 1.2.0 (optional, for better formatted output tables)
  - Download: `wget -P lib https://repo1.maven.org/maven2/com/github/freva/ascii-table/1.2.0/ascii-table-1.2.0.jar`
  - Or manually download and place in `lib/` folder

## Submission Deliverables
1. **features.csv** - Final feature file (from best performing part, found in `results/` folder)
2. **Report** - Results for all 5 parts including:
   - Part 1: Baseline accuracy with complete dataset
   - Part 2: Accuracy for each window size (1s, 2s, 3s, 4s)
   - Part 3: Accuracy with expanded features (12 features)
   - Part 4: Selected features and accuracy progression (Decision Tree)
   - Part 5: Selected features and accuracy progression (Random Forest & SVM)
   - Conclusion: Best classifier and configuration

## Output Organization
All results are organized in the `results/` folder:
- `results/part1/` - Baseline with 1s windows, 6 features
- `results/part2/` - Window tuning results (1s, 2s, 3s, 4s)
- `results/part3/` - Feature expansion with 12 features
- `results/part4/` - Feature selection (Decision Tree)
- `results/part5/` - Classifier comparison (Random Forest & SVM)
   - Conclusion: Best classifier and configuration

## Notes
- Raw data files use naming convention: `WatchID-AssignmentX-Subject-Hand-Activity-Info-DateTime.csv`
- Activity labels extracted from filenames: `hand_wash`, `non_hand_wash`
- Classification uses 10-fold cross-validation for accuracy evaluation
- Sequential Feature Selection uses forward selection (greedy approach)

### Java Reflection Warnings (Parts 4 & 5)
Parts 4 and 5 require access to internal Weka classes that use Java reflection. On Java 9+, the JVM's module system restricts this access by default, which may trigger warnings like:

```
WARNING: Illegal reflective access by weka...
```

**These warnings do NOT affect execution or results** - your program will run successfully regardless. However, to suppress these warnings and run cleanly, use the `--add-opens` flags shown in the "Run Individual Parts" section above. These flags explicitly grant Weka permission to access the required internal Java modules.

## Date
October 2025
