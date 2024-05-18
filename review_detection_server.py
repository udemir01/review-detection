import joblib
import spacy
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

model_doc2vec = joblib.load("models/sentiment_model_doc2vec.sav")
model_tfidf = joblib.load("models/sentiment_model_tfidf.sav")
model_clf = joblib.load("models/sentiment_model_clf.sav")

review_pos = "Great place to stay and many business near by."
rating_pos = 10.0
review_neg = "I won't go into details but basically employees at this hotel enforce policies in different ways. If one tells you something is okay to do, but the next person working disagrees....it will coat you the customer for there disorganization. When I asked if they expected me to check with every employee on how and which policies they enforced, they simply said yes. That is a burden no customer should bear and no customer should pay for employee disorganization."
rating_neg = 1.0


def clean_text_single(text):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    tokens = nlp(text)
    tokens = [
        token.lemma_.lower()
        for token in tokens
        if token.is_alpha
        and not token.is_stop
        and not token.is_space
    ]
    tokens = [t for t in tokens if len(t) > 1]
    text = " ".join(tokens)
    return text


def generate_features(review, rating):
    data_review = pd.DataFrame([[review, rating]], columns=["text", "rating"])
    data_review["is_negative"] = data_review["rating"].apply(lambda x: 1 if x < 5 else 0)
    data_review["text_clean"] = data_review["text"].apply(clean_text_single)
    data_review["sentiments"] = data_review["text"].apply(SentimentIntensityAnalyzer().polarity_scores)
    data_review = pd.concat([data_review.drop(["sentiments"], axis=1), data_review["sentiments"].apply(pd.Series)], axis=1)
    data_review["nb_chars"] = data_review["text"].apply(lambda x: len(x))
    data_review["nb_words"] = data_review["text"].apply(lambda x: len(x.split(" ")))
    doc2vec_df = data_review["text_clean"].apply(lambda x: model_doc2vec.infer_vector(x.split(" "))).apply(pd.Series)
    doc2vec_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df.columns]
    data_review = pd.concat([data_review, doc2vec_df], axis=1)
    tfidf_result = model_tfidf.transform(data_review["text_clean"]).toarray()
    tfidf_df = pd.DataFrame(tfidf_result, columns=model_tfidf.get_feature_names_out())
    tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
    tfidf_df.index = data_review.index
    data_review = pd.concat([data_review, tfidf_df], axis=1)
    label = "is_negative"
    ignore_cols = [label, "text", "text_clean", "rating"]
    features = [c for c in data_review.columns if c not in ignore_cols]
    return data_review[features]


def main():
    features = generate_features(review_pos, rating_pos)
    prediction = model_clf.predict(features)
    if prediction == 1:
        print(False)
    else:
        print(True)

main()
