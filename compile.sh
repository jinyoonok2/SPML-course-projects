#!/bin/bash

# compile.sh
# Removes all .class files and recompiles all Java files in the current directory

echo "====================================="
echo "Cleaning and Compiling Java Project"
echo "====================================="
echo ""

# Step 1: Remove existing .class files
echo "Step 1: Removing existing .class files..."
if ls *.class 1> /dev/null 2>&1; then
    rm -f *.class
    echo "✓ Removed all .class files"
else
    echo "✓ No .class files found (clean directory)"
fi
echo ""

# Step 2: Check for Weka library
echo "Step 2: Checking for Weka library..."
WEKA_PATH=""
if [ -f "lib/weka.jar" ]; then
    WEKA_PATH="lib/weka.jar"
    echo "✓ Found Weka at lib/weka.jar"
elif [ -f "weka.jar" ]; then
    WEKA_PATH="weka.jar"
    echo "✓ Found Weka at weka.jar"
else
    echo "✗ Error: weka.jar not found"
    echo "  Please place weka.jar in lib/ or current directory"
    exit 1
fi
echo ""

# Step 3: Count Java files
JAVA_FILES=$(ls *.java 2>/dev/null | wc -l)
if [ "$JAVA_FILES" -eq 0 ]; then
    echo "✗ No Java files found in current directory"
    exit 1
fi
echo "Step 3: Found $JAVA_FILES Java file(s) to compile"
echo ""

# Step 4: Compile all Java files
echo "Step 4: Compiling Java files..."
echo "Command: javac -cp \".:$WEKA_PATH\" *.java"
echo ""

javac -cp ".:$WEKA_PATH" *.java

# Check compilation status
if [ $? -eq 0 ]; then
    echo ""
    echo "====================================="
    echo "✓ Compilation successful!"
    echo "====================================="
    echo ""
    CLASS_FILES=$(ls *.class 2>/dev/null | wc -l)
    echo "Generated $CLASS_FILES .class file(s)"
    echo ""
    echo "You can now run the program:"
    echo "  java -cp \".:$WEKA_PATH\" Assignment2_Handler <part>"
    echo ""
else
    echo ""
    echo "====================================="
    echo "✗ Compilation failed!"
    echo "====================================="
    echo ""
    echo "Please check the error messages above and fix the issues."
    exit 1
fi
