# ASTC Dataset
The Arabic Sentiment Twitter Corpus (ASTC) Corpus contains 58K Arabic tweets (47K training, 11K test) tweets annotated in positive and negative labels. The dataset is balanced and collected using positive and negative emojis lexicon.

## Performance evaluation
### Bag of words
| Model| Accuracy (%) | Percision (%) | Recall (%) | F1 score (%)|
| -| - | - | - | - |
| NB | 61.63 | 68.27 | 40.32 | 50.70 |
| Logistic regression | 90 | 86.49 | 92.77 | 89.52 |
| SVM | 58.45 | 72.61 | 24.20 | 36.31 |
| Random Forest | 88.87 | 86.01 | 92.25 | 89.02 |

### TF-IDF
| Model| Accuracy (%) | Percision (%) | Recall (%) | F1 score (%)|
| -| - | - | - | - |
| NB | 62.78 | 65.68 | 50.10 | 56.85 |
| Logistic regression | 89.75 | 86.63 | 93.46 | 89.92 |
| SVM | 54.10 | 65.30 | 13.21 | 13.21 |
| Random Forest | 77.31 | 82.12 | 69.66 | 75.38 |
### LDA
| Model| Accuracy (%) | Percision (%) | Recall (%) | F1 score (%)|
| -| - | - | - | - |
| NB | 70.87 | 77.21 | 58.99 | 66.88 |
| Logistic regression | 72.27 | 74.53 | 67.42 | 70.80 |
| SVM | 71.99 | 75.32 | 65.17 | 69.88 |
| Random Forest | 66.39 | 64.95 | 70.79 | 67.74 |
### LSA
| Model| Accuracy (%) | Percision (%) | Recall (%) | F1 score (%)|
| -| - | - | - | - |
| NB | 54.9 | 52.63 | 95.51 | 67.86 |
| Logistic regression | 78.71 | 82.28 | 73.03 | 77.38 |
| SVM | 76.19 | 81.21 | 67.98 | 74.01 |
| Random Forest | 79.83 | 79.44 | 80.34 | 79.89 |
### BoC
| Model| Accuracy (%) | Percision (%) | Recall (%) | F1 score (%)|
| -| - | - | - | - |
| NB | 62.18 | 90.57 | 26.97 | 41.56 |
| Logistic regression | 80.39 | 81.40 | 78.65 | 80 |
| SVM | 75.35 | 75.86 | 74.16 | 75 |
| Random Forest | 73.67 | 74.14 | 72.47 | 73.3 |