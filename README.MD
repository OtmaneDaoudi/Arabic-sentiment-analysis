# LABR: A Large-SCale Arabic Book Reviews Dataset
This dataset contains over 63,000 book reviews in Arabic. It is the largest sentiment analysis dataset for Arabic to date. The book reviews were harvested from the website Goodreads in March 2013.

## Performance evaluation
### Bag of words
| Model| Accuracy (%) | Precision (%) | Recall (%) | F1 score (%)|
| -| - | - | - | - |
| NB | 38.07 | 88.90 | 32.82 | 47.94 |
| Logistic regression | 87.64 | 89.92 | 96.60 | 93.14 |
| SVM | 86.88 | 86.88 | 86.88 | 86.88 |
| Random Forest | 87.59 | 87.59 | 87.59 | 87.59 |

### TF-IDF
| Model| Accuracy (%) | Precision (%) | Recall (%) | F1 score (%)|
| -| - | - | - | - |
| NB | 38.78 | 88.94 | 33.73 | 48.91 |
| Logistic regression | 88.59 | 89.12 | 98.95 | 93.78 |
| SVM | 87 | 87 | 87 | 87 |
| Random Forest | 87.6 | 87.6 | 87.6 | 87.6 |
### LDA
| Model| Accuracy (%) | Precision (%) | Recall (%) | F1 score (%)|
| -| - | - | - | - |
| NB | 51.53 | 91.78 | 48.56 | 63.52 |
| Logistic regression | 86.89 | 86.89 | 100 | 92.98 |
| SVM | 86.88 | 86.88 | 86.88 | 86.88 |
| Random Forest | 86.61 | 86.61 | 86.61 | 86.61 |
### LSA
| Model| Accuracy (%) | Precision (%) | Recall (%) | F1 score (%)|
| -| - | - | - | - |
| NB | 31.80 | 88.83 | 24.60 | 38.53 |
| Logistic regression | 86.89 | 87.40 | 99.22 | 92.94 |
| SVM | 86.88 | 86.88 | 86.88 | 86.88 |
| Random Forest | 87 | 87 | 87 | 87 |
### BoC
| Model| Accuracy (%) | Precision (%) | Recall (%) | F1 score (%)|
| -| - | - | - | - |
| NB | 16.61 | 91.67 | 4.42 | 4.42 |
| Logistic regression | 86.84 | 86.98 | 99.80 | 92.95 |
| SVM | 86.84 | 86.84 | 86.84 | 86.84 |
| Random Forest | 86.69 | 86.69 | 86.69 | 86.69 |
