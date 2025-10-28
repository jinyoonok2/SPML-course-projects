#!/bin/bash

# check_jdk.sh
# Checks if JDK is installed and displays version information

echo "====================================="
echo "Checking Java Development Kit (JDK)"
echo "====================================="
echo ""

# Check if javac is available
if command -v javac &> /dev/null
then
    echo "✓ JDK is installed"
    echo ""
    echo "Java Compiler Version:"
    javac -version
    echo ""
    echo "Java Runtime Version:"
    java -version
    echo ""
    
    # Check Java version number
    JAVA_VERSION=$(javac -version 2>&1 | awk '{print $2}')
    MAJOR_VERSION=$(echo $JAVA_VERSION | cut -d'.' -f1)
    
    if [ "$MAJOR_VERSION" -ge 8 ]; then
        echo "✓ Java version is compatible (JDK 8 or higher required)"
    else
        echo "✗ Warning: JDK 8 or higher is recommended"
        echo "  Current version: $JAVA_VERSION"
    fi
else
    echo "✗ JDK is NOT installed"
    echo ""
    echo "Please install JDK 8 or higher:"
    echo "  - Ubuntu/Debian: sudo apt install default-jdk"
    echo "  - Fedora/RHEL: sudo dnf install java-devel"
    echo "  - Arch: sudo pacman -S jdk-openjdk"
    exit 1
fi

echo ""
echo "====================================="
echo "Checking for required libraries"
echo "====================================="
echo ""

# Check if weka.jar exists
if [ -f "lib/weka.jar" ]; then
    echo "✓ Weka library found at lib/weka.jar"
elif [ -f "weka.jar" ]; then
    echo "✓ Weka library found at weka.jar"
else
    echo "✗ Weka library NOT found"
    echo ""
    echo "Please download Weka and place weka.jar in:"
    echo "  - lib/weka.jar (recommended)"
    echo "  - or current directory: weka.jar"
    echo ""
    echo "Download from: https://www.cs.waikato.ac.nz/ml/weka/"
    exit 1
fi

# Check if ascii-table.jar exists
if [ -f "lib/ascii-table-1.2.0.jar" ]; then
    echo "✓ ASCII Table library found at lib/ascii-table-1.2.0.jar"
else
    echo "⚠ ASCII Table library NOT found (optional)"
    echo ""
    echo "For better formatted tables, download:"
    echo "  wget -P lib https://repo1.maven.org/maven2/com/github/freva/ascii-table/1.2.0/ascii-table-1.2.0.jar"
    echo ""
fi

echo ""
echo "====================================="
echo "All checks passed! Ready to compile."
echo "====================================="
