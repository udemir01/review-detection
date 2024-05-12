import spacy as sp
import pandas as pd


def text_processing(df):
    nlp = sp.load("en_core_web_sm")
    pos_doc = df["PositiveReview"].astype(str).unique()
    neg_doc = df["NegativeReview"].astype(str).unique()
    for i, text in enumerate(pos_doc):
        doc = nlp(text.lower())
        doc = list(token.lemma_ for token in doc if not token.is_stop)
        doc = list(token.vector for token in doc if not token.is_stop)
        print(doc)
        pos_doc[i] = doc
    for i, text in enumerate(neg_doc):
        doc = nlp(text.lower())
        doc = list(token.lemma_ for token in doc if not token.is_stop)
        doc = list(token.vector for token in doc if not token.is_stop)
        print(doc)
        neg_doc[i] = doc


def main():
    sp.prefer_gpu()
    df = pd.read_csv("dataset/booking_hotel.csv")
    text_processing(df[["PositiveReview", "NegativeReview"]])


main()
