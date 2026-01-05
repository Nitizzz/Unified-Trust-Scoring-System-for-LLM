# Debugging Code Hallucination Head Results

## Changes Made
- **Data Loading**: Optimized tensor creation using `numpy` to eliminate performance warnings.
- **Features (`features.py`)**: Enhanced `mutate_code` to ensure synthetic mutations always produce semantically different code (avoiding "no-op" mutations).
- **Training (`train.py`)**:
    - Implemented **GroupShuffleSplit** to keep variants of the same question together (preventing data leakage).
    - Added **Class Weights** (`pos_weight`) to the loss function to handle imbalance (Code Hallucinations are ~33% of the dataset).
    - Added detailed execution logging (Confusion Matrix, Probability Stats) and file output.

## Validation Results
We ran the training loop for 10 epochs on the generated dataset (150 samples).

### Metrics Comparison
| Metric | Before Fix | After Fix |
| :--- | :--- | :--- |
| **Code Hallucination F1** | ~0.000 | **0.800** |
| **Code Precision** | ~0.000 | **0.800** |
| **Code Recall** | ~0.000 | **0.800** |
| **Trust Score Correlation** | ~0.500 | **0.768** |

### Confusion Matrix (Test Set, 30 Samples)
```
[[18  2]   <-- True Negatives (Faithful), False Positives
 [ 2  8]]  <-- False Negatives, True Positives (Hallucinated)
```
The model correctly identified 8 out of 10 hallucinated code snippets, with only 2 false positives.

### Conclusion
The "Code Hallucination Head" is now functional and contributing significantly to the overall Trust Score correlation (0.768). The 0-F1 issue was resolved by a combination of better synthetic data generation and proper handling of class imbalance.
