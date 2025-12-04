#!/usr/bin/env python3
"""
Confusion Matrix Visualization Script
Reads confusion matrix CSV files and generates heatmap images
"""

import sys
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_confusion_matrix(csv_file, output_file=None):
    """
    Read confusion matrix from CSV and generate heatmap image
    
    Args:
        csv_file: Path to CSV file with confusion matrix
        output_file: Path to save PNG image (optional, auto-generated if None)
    """
    # Read CSV file
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    # First row is header with class labels
    class_labels = rows[0][1:]  # Skip first column header
    
    # Parse confusion matrix data
    matrix = []
    row_labels = []
    for row in rows[1:]:
        row_labels.append(row[0])
        matrix.append([float(x) for x in row[1:]])
    
    matrix = np.array(matrix)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create heatmap with annotations
    sns.heatmap(matrix, annot=True, fmt='.0f', cmap='Blues',
                xticklabels=class_labels, yticklabels=row_labels,
                cbar_kws={'label': 'Count'}, linewidths=0.5, linecolor='gray')
    
    plt.xlabel('Predicted Class', fontsize=12, fontweight='bold')
    plt.ylabel('Actual Class', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Auto-generate output filename if not provided
    if output_file is None:
        output_file = str(csv_file).replace('.csv', '.png')
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved confusion matrix image: {output_file}")
    plt.close()


def plot_all_confusion_matrices(directory):
    """
    Find and plot all confusion matrix CSV files in a directory tree
    
    Args:
        directory: Root directory to search for confusion_matrix.csv files
    """
    csv_files = list(Path(directory).rglob('*confusion_matrix.csv'))
    
    if not csv_files:
        print(f"No confusion matrix CSV files found in {directory}")
        return
    
    print(f"Found {len(csv_files)} confusion matrix file(s)")
    
    for csv_file in csv_files:
        try:
            plot_confusion_matrix(csv_file)
        except Exception as e:
            print(f"✗ Error processing {csv_file}: {e}")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python plot_confusion_matrix.py <csv_file>         # Plot single file")
        print("  python plot_confusion_matrix.py <directory>        # Plot all CSV files in directory")
        print()
        print("Examples:")
        print("  python plot_confusion_matrix.py results/baseline/j48/confusion_matrix.csv")
        print("  python plot_confusion_matrix.py results/")
        sys.exit(1)
    
    path = sys.argv[1]
    
    if os.path.isfile(path):
        # Single file mode
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        plot_confusion_matrix(path, output_file)
    elif os.path.isdir(path):
        # Directory mode - find all confusion matrix CSVs
        plot_all_confusion_matrices(path)
    else:
        print(f"Error: {path} is not a valid file or directory")
        sys.exit(1)


if __name__ == '__main__':
    main()
