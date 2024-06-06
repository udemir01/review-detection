import joblib
import spacy
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flask import Flask, request, jsonify
from flask_cors import cross_origin

model_tfidf_sentiment = joblib.load("models/sentiment_model_tfidf.sav")
model_clf_sentiment = joblib.load("models/sentiment_model_clf.sav")
model_tfidf_fakereview = joblib.load("models/fakereview_model_tfidf.sav")
model_clf_fakereview = joblib.load("models/fakereview_model_clf.sav")

app = Flask(__name__)


def check_pos_tag(token):
    if token.pos_ == "PROPN":
        return False
    elif token.pos_ == "DET":
        return False
    elif token.pos_ == "PRON":
        return False
    elif token.pos_ == "AUX":
        return False
    else:
        return True


def clean_text_single(text):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    tokens = nlp(text)
    tokens = [
        token.lemma_.lower()
        for token in tokens
        if token.is_alpha
        and not token.is_stop
        and check_pos_tag(token)
    ]
    tokens = [t for t in tokens if len(t) > 1]
    text = " ".join(tokens)
    return text


def generate_features(review, rating, tfidf):
    data_review = pd.DataFrame([[review, rating]], columns=["text", "rating"])
    data_review["text_clean"] = data_review["text"].apply(clean_text_single)
    data_review["sentiments"] = data_review["text"].apply(SentimentIntensityAnalyzer().polarity_scores)
    data_review = pd.concat([data_review.drop(["sentiments"], axis=1), data_review["sentiments"].apply(pd.Series)], axis=1)
    data_review["nb_chars"] = data_review["text"].apply(lambda x: len(x))
    data_review["nb_words"] = data_review["text"].apply(lambda x: len(x.split(" ")))
    tfidf_result = tfidf.transform(data_review["text_clean"])
    tfidf_df = pd.DataFrame(tfidf_result.toarray(), columns=tfidf.get_feature_names_out())
    tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
    tfidf_df.index = data_review.index
    data_review = pd.concat([data_review, tfidf_df], axis=1)
    ignore_cols = ["text", "text_clean", "rating"]
    features = [c for c in data_review.columns if c not in ignore_cols]
    return data_review[features]


@app.route("/detection_server", methods=["POST"])
@cross_origin(origin="*", headers=["Content-Type", "Authorization"])
def main():
    data = request.get_json()
    review = data.get("description", "")
    rating = data.get("userScore", "")
    features_sentiment = generate_features(review, rating, model_tfidf_sentiment)
    features_fakereview = generate_features(review, rating, model_tfidf_fakereview)
    prediction_sentiment = model_clf_sentiment.predict(features_sentiment)
    prediction_fakereview = model_clf_fakereview.predict(features_fakereview)
    if prediction_fakereview == 1:
        return jsonify({"result": False})
    else:
        return jsonify({"result": True})


if __name__ == "__main__":
    app.run(port=5000, debug=True)
