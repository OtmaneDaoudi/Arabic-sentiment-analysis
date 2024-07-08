# Project overview
The aim of this project is to create an arabic sentiment analysis system, that takes adantage of the different text representation models like TF-IDF, Bag of words, Bag of concepts in addition to exploring newer methods such as appraisal theory.

# Architecture
The general architecture of the system is the following:

# Repository layout
The code for the system is organized into the following branches:
1. *AJGT*: Code for arabic sentiment analysis using classical machine learning models built using the AJGT dataset.
2. *ASTC*: Code for arabic sentiment analysis using classical machine learning models built using the ASTC dataset.
3. *ASTD*: Code for arabic sentiment analysis using classical machine learning models built using the ASTC dataset.
4. *LABR*: Code for arabic sentiment analysis using classical machine learning models built using the ASTC dataset.
5. *DL*: Code for arabic sentiment analysis using deep learning.
6. *Appraisal*: Code for arabic sentiment analysis using appraisal features.
7. *Deployment*: Deployment of the system using streamlit.

# Datasets
- LABR (Large scale Arabic Book Reviews)
- AJGT (The Arabic Jordanian General Tweets)
- ASTC (Arabic Sentiment Twitter Corpus)
- ASTD (Arabic Sentiment Tweets Dataset)

# Text representation models
The system supports the following text representation models:
- Bow (Bag of Words)
- TF-IDF (Term frequency, inverse document frequency)
- LSA (Latent semantic analysis)
- LDA (Latent ddirechlet allocation)
- BoC (Bag of Concepts)
- Appraisal groups

# Machine learning models
The previous text representation modes are used to create features for the following models:
- Naive bayes
- Logistic regression
- Support Vector Machine
- Random forest

As for deep learning, we opted for the BERT (Bidirectional Encoder Representations) model and its variants.