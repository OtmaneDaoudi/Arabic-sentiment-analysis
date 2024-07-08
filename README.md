# Project overview
This project aims to create an Arabic sentiment analysis system, that takes advantage of the different text representation models like TF-IDF, Bag of words, and Bag of concepts in addition to exploring newer methods such as appraisal theory.

# Architecture
The general architecture of the system is the following:

![image](https://github.com/OtmaneDaoudi/Arabic-sentiment-analysis/assets/63020343/82a8d42c-b068-4c35-b467-932b72a208f1)


# Repository layout
The code for the system is organized into the following branches:
1. [*AJGT*](https://github.com/OtmaneDaoudi/Arabic-sentiment-analysis/tree/AJGT): Code for Arabic sentiment analysis using classical machine learning models built using the AJGT dataset.
2. [*ASTC*](https://github.com/OtmaneDaoudi/Arabic-sentiment-analysis/tree/ASTC): Code for Arabic sentiment analysis using classical machine learning models built using the ASTC dataset.
3. [*ASTD*](https://github.com/OtmaneDaoudi/Arabic-sentiment-analysis/tree/ASTD): Code for Arabic sentiment analysis using classical machine learning models built using the ASTC dataset.
4. [*LABR*](https://github.com/OtmaneDaoudi/Arabic-sentiment-analysis/tree/LABR): Code for Arabic sentiment analysis using classical machine learning models built using the ASTC dataset.
5. [*DL*](https://github.com/OtmaneDaoudi/Arabic-sentiment-analysis/tree/DL): Code for Arabic sentiment analysis using deep learning.
6. [*Appraisal*](https://github.com/OtmaneDaoudi/Arabic-sentiment-analysis/tree/appraisal): Code for Arabic sentiment analysis using appraisal features.
7. [*Deployment*](https://github.com/OtmaneDaoudi/Arabic-sentiment-analysis/tree/Deployment): Deployment of the system using Streamlit.

Each branch contains a details overview of the dataset used, as well as all the performance metrics.

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
- LDA (Latent Dirichlet allocation)
- BoC (Bag of Concepts)
- Appraisal groups

# Machine learning models
The previous text representation modes are used to create features for the following models:
- Naive bayes
- Logistic regression
- Support Vector Machine
- Random forest

As for deep learning, we opted for the BERT (Bidirectional Encoder Representations) model and its variants.
