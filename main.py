import re
import pickle

import streamlit as st

from snowballstemmer import stemmer
from pyarabic.araby import tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from appraisal import AppraisalUtils


def preprocess(text: str) -> str:
    # Normalize hamza
    pattern = r"أ|إ|آ"
    text = re.sub(pattern, 'ا', text)

    # Tokenization
    text = tokenize(text)

    # # Removing long and short words
    text = [word for word in text if len(word) < 9 or len(word) > 1]

    # Stemming
    ar_stemmer = stemmer("arabic")
    text = [ar_stemmer.stemWord(word) for word in text]

    # remove stopwords
    stopwords = pickle.loads(open("./models/stopwords.pkl", 'rb').read())
    text = [word for word in text if word not in stopwords]

    return " ".join(text)


def evaluate() -> str:
    text = document
    text = preprocess(text)
    print("preprocessed text : ", text)

    repres_to_dir = {
        "Bag-of-words": "BOW",
        "TF-IDF": "TFIDF",
        "LDA": "LDA",
        "LSA": "LSA",
        "Bag-Of-Concepts": "BOW",
        "Bag-of-words w/ Appraisal": "APPRAISAL"
    }

    model_to_dir = {
        "Naive bayes": "NB",
        "Logistic regression": "RF",
        "SVM": "SVM_Pipeline",
        "Random forest": "RF"
    }

    vectorizer_path = f"./models/ASTC/{repres_to_dir[representataion]}/vectorizer.pkl"
    vectorizer: CountVectorizer | TfidfVectorizer = pickle.loads(
        open(vectorizer_path, 'rb').read())

    model_path = f"./models/ASTC/{repres_to_dir[representataion]}/{model_to_dir[model]}.pkl"
    model_: LogisticRegression = pickle.loads(
        open(model_path, 'rb').read())

    x = vectorizer.transform([text])
    if representataion in ["Bag-of-words", "TF-IDF"]:
        x = x.toarray()
    st.session_state["result"] = model_.predict(x)[0]


"""
# Arabic sentiment analysis
"""
document = st.text_input(label="Text to analyze :")
representataion = st.selectbox(label="Text representation", options=[
    "Bag-of-words",
    "Bag-of-words w/ Appraisal",
    "TF-IDF",
    "LDA",
    "LSA",
    "Bag-Of-Concepts"
])

model = st.selectbox(label="Model", options=[
    "Naive bayes",
    "Logistic regression",
    "SVM",
    "Random forest"
])

btn = st.button(label="Evaluate", on_click=evaluate)

try:
    if st.session_state.result:
        if st.session_state.result == "pos":
            st.markdown(
                "<h2 style='text-align: center; color: lightgreen;'> Positive 😄 </h2>", unsafe_allow_html=True)
            st.session_state["result"] = None
        else:
            st.markdown(
                "<h2 style='text-align: center; color: #d4311c;'> Negative 😔 </h2>", unsafe_allow_html=True)
            st.session_state["result"] = None
except:
    pass
