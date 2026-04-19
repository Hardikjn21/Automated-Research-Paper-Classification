# Multi-Label Text Classification using Transformer Models (DeBERTa, RoBERTa)

## Overview

This project implements a multi-label text classification system for scientific articles using transformer-based models. Each input consists of a paper’s **title and abstract**, and the goal is to predict one or more subject categories (57 total labels).

The solution leverages:

* Transformer models (DeBERTa-base, DeBERTa-large, RoBERTa)
* Custom loss function for class imbalance (ResampleLoss)
* Ensemble learning for improved performance
* Threshold optimization for better F1 score

---

## Problem Statement

Given a dataset of research papers with titles and abstracts, classify each paper into one or more predefined categories such as:

* cs.AI, cs.CV, cs.LG
* math.ST, math.PR
* stat.ML, etc.

This is a **multi-label classification problem**, meaning:

* Each sample can belong to multiple categories
* Labels are not mutually exclusive

---

## Dataset

### Input Features

* `Title`
* `Abstract`
* Combined into a single text field: `combined = Title + Abstract`

### Target Labels

* 57 categories (excluding metadata columns)
* Each label is binary (0 or 1)

---

## Project Architecture

### 1. Data Processing

* Combine title and abstract
* Tokenize text using model-specific tokenizers
* Convert labels into multi-hot encoded vectors

### 2. Dataset Class

Custom PyTorch dataset:

* Tokenizes input text
* Returns:

  * input_ids
  * attention_mask
  * token_type_ids
  * targets

---

## Models Used

### DeBERTa Base

* Hidden size: 768
* 12 transformer layers

### DeBERTa Large

* Hidden size: 1024
* 24 transformer layers

### RoBERTa Base

* Hidden size: 768
* Uses pooled output instead of CLS token

---

## Model Architecture

Each model follows:

1. Input text → Transformer
2. Extract representation:

   * DeBERTa: CLS token (`last_hidden_state[:, 0]`)
   * RoBERTa: `pooler_output`
3. Apply Dropout (0.3)
4. Linear layer:

   * Input: hidden size (768 or 1024)
   * Output: 57 logits

---

## Loss Function: ResampleLoss

A custom loss function designed to handle:

* Class imbalance
* Multi-label structure
* Dominance of negative labels

### Components:

1. Reweighting

   * Uses inverse class frequency
   * Normalizes using only positive labels (repeat_rate)

2. Logit Regulation

   * Scales negative logits to reduce their dominance

3. Binary Cross Entropy

   * Base loss for multi-label classification

4. Focal Loss

   * Focuses on hard examples
   * Reduces importance of easy predictions

Final Loss:

```
Loss = Weight × BCE × Focal Factor
```

---

## Training Process

For each batch:

1. Pass inputs through model → logits
2. Compute loss using ResampleLoss
3. Backpropagation:

   * Compute gradients
   * Update model weights
4. Track:

   * Loss
   * Accuracy

---

## Evaluation

* Model is evaluated on validation and test sets
* Uses:

  * Accuracy (element-wise)
  * F1 Score (macro)

---

## Threshold Optimization

Instead of using a fixed threshold (0.5), the system searches for an optimal threshold.

### Process:

1. Try thresholds from 0.05 to 0.65
2. Convert probabilities to predictions
3. Compute F1 score
4. Select threshold with highest F1

This improves balance between:

* Precision
* Recall

---

## Ensemble Method

Four models are used together:

* DeBERTa Base (2 variants)
* DeBERTa Large
* RoBERTa Base

### Process:

1. Each model produces logits
2. Outputs are averaged:

   ```
   combined = (model1 + model2 + model3 + model4) / 4
   ```
3. Apply sigmoid → probabilities
4. Apply threshold → predictions

### Benefit:

* Reduces variance
* Improves generalization
* More stable predictions

---

## Post-processing

### Handling All-Zero Predictions

Problem:

* Sometimes model predicts no labels

Solution:

* Assign the label with highest probability

---

## Output Format

Final predictions are:

* Converted into a DataFrame
* Columns match required submission format
* Saved as CSV

---

## Complete Pipeline

1. Load test data
2. Preprocess text
3. Tokenize using multiple tokenizers
4. Pass through multiple models
5. Average predictions (ensemble)
6. Apply sigmoid and threshold
7. Fix edge cases
8. Format output
9. Save submission file

---

## Key Insights

* Multi-label problems require different handling than single-label classification
* Class imbalance significantly affects learning
* Combining models (ensemble) improves robustness
* Threshold tuning is critical for performance
* Custom loss functions can significantly improve results

---

## Possible Improvements

* Per-label threshold optimization
* Weighted ensemble instead of simple average
* Use of more diverse transformer architectures
* Hyperparameter tuning (learning rate, dropout)
* Incorporating metadata features

---

## Dependencies

* PyTorch
* Transformers (HuggingFace)
* NumPy
* Pandas
* scikit-learn
* tqdm

---

## Conclusion

This project builds a robust multi-label classification system using advanced techniques:

* Transformer-based models
* Custom loss for imbalance
* Ensemble learning
* Threshold optimization

The combination of these techniques results in improved performance and reliable predictions for complex multi-label tasks.
