# Legacy Code Archive

This folder contains the original Part-based implementation that has been replaced by the new engine-based architecture.

## Archived Files

- **Part1_DataProcessor.java** - Original baseline experiments (replaced by DataManager + FeatureEngine + MLEngine)
- **Part2_WindowTuning.java** - Original window optimization (replaced by OptimizationEngine)
- **Part3_FeatureExpansion.java** - Original feature expansion (replaced by FeatureEngine)
- **Part4_FeatureSelection.java** - Original feature selection (replaced by OptimizationEngine)
- **Part5_ClassifierComparison.java** - Original classifier comparison (replaced by MLEngine)
- **WadaManager.java** - Original workflow manager (replaced by new WadaManager + ExperimentRunner)

## Why These Were Archived

The original Part-based code was replaced with a new engine-based architecture that provides:

1. **Better organization** - Code grouped by responsibility instead of arbitrary "parts"
2. **Eliminated duplication** - Feature extraction code was duplicated across Parts 1-3
3. **Clean separation of concerns** - Each engine has a single, well-defined purpose
4. **Enhanced maintainability** - Easier to understand, test, and modify
5. **Improved usability** - Modern streamlined commands + backward compatibility

## Code Reuse

**The new architecture reuses ~90% of the actual implementation code** from these files:
- All algorithms, formulas, and calculations are preserved
- Feature extraction logic consolidated from Parts 1-3
- Window optimization logic from Part 2
- Sequential Feature Selection from Part 4
- Classifier comparison from Part 5

## New Architecture

The functionality has been redistributed into specialized engines:

```
DataManager        → Raw data processing (from Part1)
FeatureEngine      → Feature extraction (from Part1, Part2, Part3)
OptimizationEngine → Parameter optimization (from Part2, Part4)
MLEngine           → ML operations (from Part5 + orchestration)
ExperimentRunner   → New unified command interface
WadaManager        → Updated to use new engines (backward compatible)
```

## Backward Compatibility

The new system maintains full backward compatibility:
```bash
# Old commands still work
java WadaManager part1
java WadaManager all

# New streamlined commands available
java ExperimentRunner baseline
java ExperimentRunner experiment
```

## Recovery

If needed, these files can be restored from this archive or from git history:
```bash
# Restore from archive
cp legacy/Part1_DataProcessor.java .

# Or from git
git checkout HEAD -- Part1_DataProcessor.java
```

---
**Archived on:** December 3, 2025
**Reason:** Replaced by new engine-based architecture
**Status:** Fully functional but superseded by better design
