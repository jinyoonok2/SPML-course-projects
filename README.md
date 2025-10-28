# SPML Course - Assignment 2: Gesture Recognition Optimization

## Overview
This assignment builds upon Assignment 1 to improve gesture recognition accuracy through parameter tuning, feature engineering, and classifier comparison. The project uses smartwatch accelerometer data to classify hand-washing gestures.

## Project Structure

```
SPML-course-projects/
├── raw_data/                           # All raw CSV files from smartwatch (Assignment 1 + 2)
│   └── G5NZCJ022402200-Assignment*-Jinyoon-*-hand_wash-*.csv
├── formatted_data/                     # Cleaned data (timestamp, ax, ay, az)
├── lib/
│   └── weka.jar                        # Weka library (download separately)
├── features.csv                        # Generated features with labels
├── features.arff                       # Weka-compatible feature file
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

### Part 3: Feature Expansion
- **Goal**: Add median and RMS (Root Mean Square) features
- **Implementation**: `Part3_FeatureExpansion.java`
- **Features**: 12 features (mean, std, median, RMS per axis)
- **Window**: Best window from Part 2
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
./compile.sh
```
This script:
- Removes all existing .class files
- Recompiles all Java files in the directory
- Verifies successful compilation

### Run Individual Parts
```bash
# Part 1: Data combination
java -cp ".:lib/weka.jar" Assignment2_Handler part1

# Part 2: Window tuning
java -cp ".:lib/weka.jar" Assignment2_Handler part2

# Part 3: Feature expansion
java -cp ".:lib/weka.jar" Assignment2_Handler part3

# Part 4: Feature selection (Decision Tree)
java -cp ".:lib/weka.jar" Assignment2_Handler part4

# Part 5: Classifier comparison (Random Forest & SVM)
java -cp ".:lib/weka.jar" Assignment2_Handler part5
```

### Run All Parts
```bash
java -cp ".:lib/weka.jar" Assignment2_Handler all
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

## Dependencies
- **Java**: JDK 8 or higher
- **Weka**: 3.8.x or higher (place `weka.jar` in `lib/` directory)
- Download Weka from: https://www.cs.waikato.ac.nz/ml/weka/

## Submission Deliverables
1. **features.csv** - Final feature file
2. **Report** - Results for all 5 parts including:
   - Part 1: Accuracy with more training data
   - Part 2: Accuracy for each window size
   - Part 3: Accuracy with expanded features
   - Part 4: Selected features and accuracy progression (Decision Tree)
   - Part 5: Selected features and accuracy progression (Random Forest & SVM)
   - Conclusion: Best classifier and configuration

## Notes
- Raw data files use naming convention: `WatchID-AssignmentX-Subject-Hand-Activity-Info-DateTime.csv`
- Activity labels extracted from filenames: `hand_wash`, `non_hand_wash`
- Classification uses 10-fold cross-validation for accuracy evaluation
- Sequential Feature Selection uses forward selection (greedy approach)

## Author
Jinyoon Ok

## Date
October 2025
