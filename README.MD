This branch contains the implementation of appraisal theory based on the following paper : [Using Appraisal Groups for Sentiment Analysis](https://www.researchgate.net/publication/200044325_Using_appraisal_groups_for_sentiment_analysis)

## Performance evaluation
The results we obtained on the AJGT dataset are the following:
### G:AO
| Model| Accuracy (%) | Precision (%) | Recall (%) | F1 score (%)|
| -| - | - | - | - |
| NB | 51.14 | 50.04 | 95.30 | 65.62 |
| Logistic regression | 64.86 | 80.12 | 37.48 | 51.07 |
| SVM | 64.73 | 80.13 | 37.10 | 50.72 |
| Random Forest | 64.73 | 80.13 | 37.10 | 50.72 |
### Bag of words + G:AO
| Model| Accuracy (%) | Percision (%) | Recall (%) | F1 score (%)|
| -| - | - | - | - |
| NB | 58.53 | 55.61 | 75.55 | 64.07 |
| Logistic regression | 87.50 | 85.21 | 90.08 | 87.58 |
| SVM | 86.84 | 82.48 | 92.81 | 87.34 |
| Random Forest | 86.84 | 82.48 | 92.81 | 87.34 |


