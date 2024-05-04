# AJGT Dataset
The Arabic Jordanian General Tweets (AJGT) Corpus consists of 1,800 tweets annotated as positive and negative, in modern standard arabic (MSA) and Jordanian dialect.

## Performance evaluation
### Bag of words
| Model| Accuracy (%) | Percision (%) | Recall (%) | F1 score (%)|
| -| - | - | - | - |
| NB | 78.43 | 73.06 | 80.9 | 80.6 |
| Logistic regression | 84.59 | 86.39 | 82.02 | 84.15 |
| SVM | 82.07 | 85.62 | 76.97 | 81.07 |
| Random Forest | 78.15 | 81.25 | 73.03 | 76.92 |

### TF-IDF
| Model| Accuracy (%) | Percision (%) | Recall (%) | F1 score (%)|
| -| - | - | - | - |
| NB | 76.75 | 72.51 | 85.96 | 78.66 |
| Logistic regression | 84.03 | 86.23 | 80.90 | 83.48 |
| SVM | 85.15 | 88.34 | 80.90 | 84.46 |
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