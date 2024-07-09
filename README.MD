# ASTC Dataset
The Arabic Sentiment Twitter Corpus (ASTC) Corpus contains 58K Arabic tweets (47K training, 11K test) tweets annotated with positive and negative labels. The dataset is balanced and collected using positive and negative emojis lexicon.

## Performance evaluation
### Bag of words
| Model| Accuracy (%) | Precision (%) | Recall (%) | F1 score (%)|
| -| - | - | - | - |
| NB | 61.63 | 68.27 | 40.32 | 50.70 |
| Logistic regression | 90 | 86.49 | 92.77 | 89.52 |
| SVM | 58.45 | 72.61 | 24.20 | 36.31 |
| Random Forest | 88.87 | 86.01 | 92.25 | 89.02 |

### TF-IDF
| Model| Accuracy (%) | Precision (%) | Recall (%) | F1 score (%)|
| -| - | - | - | - |
| NB | 62.78 | 65.68 | 50.10 | 56.85 |
| Logistic regression | 89.75 | 86.63 | 93.46 | 89.92 |
| SVM | 54.10 | 65.30 | 13.21 | 13.21 |
| Random Forest | 89.04 | 86.10 | 92.53 | 89.20 |
### LDA
| Model| Accuracy (%) | Precision (%) | Recall (%) | F1 score (%)|
| -| - | - | - | - |
| NB | 61.12 | 61.14 | 56.36 | 58.65 |
| Logistic regression | 61.90 | 62.64 | 54.84 | 58.48 |
| SVM | 61.17 | 64.13 | 46.85 | 54.15 |
| Random Forest | 60.40 | 60.5 | 54.88 | 57.55 |
### LSA
| Model| Accuracy (%) | Precision (%) | Recall (%) | F1 score (%)|
| -| - | - | - | - |
| NB | 67.77 | 75.82 | 50.10 | 60.34 |
| Logistic regression | 88.67 | 85.14 | 93.08 | 88.93 |
| SVM | 87.84 | 83.32 | 93.95 | 88.31 |
| Random Forest | 88.28 | 86.86 | 89.59 | 88.20 |
### BoC
| Model| Accuracy (%) | Precision (%) | Recall (%) | F1 score (%)|
| -| - | - | - | - |
| NB | 66.54 | 69.04 | 57.30 | 62.62 |
| Logistic regression | 74.95 | 73.54 | 76.21 | 74.85 |
| SVM | 75.84 | 75.24 | 75.45 | 75.35 |
| Random Forest | 75.62 | 75.04 | 75.17 | 75.11 |
