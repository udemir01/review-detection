import joblib
import spacy
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flask import Flask, request, jsonify
from flask_cors import cross_origin
import time

model_clf_sentiment = joblib.load("models/sentiment_model_clf.sav")
model_clf_fakereview = joblib.load("models/fakereview_model_clf.sav")
model_lda_sentiment = joblib.load("models/sentiment_model_lda.sav")
model_lda_fakereview = joblib.load("models/fakereview_model_lda.sav")
model_cv_sentiment = joblib.load("models/sentiment_model_cv.sav")
model_cv_fakereview = joblib.load("models/fakereview_model_cv.sav")
model_doc2vec_sentiment = joblib.load("models/sentiment_model_doc2vec.sav")
model_doc2vec_fakereview = joblib.load("models/fakereview_model_doc2vec.sav")
model_tfidf_sentiment = joblib.load("models/sentiment_model_tfidf.sav")
model_tfidf_fakereview = joblib.load("models/fakereview_model_tfidf.sav")
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

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


def preprocess_text(data):
    data["text_clean"] = data["text"].apply(clean_text_single)
    return data


def extract_vader(data):
    data["sentiments"] = data["text"].apply(SentimentIntensityAnalyzer().polarity_scores)
    data = pd.concat([data.drop(["sentiments"], axis=1), data["sentiments"].apply(pd.Series)], axis=1)
    return data


def extract_num_char(data):
    data["nb_chars"] = data["text"].apply(lambda x: len(x))
    return data


def extract_num_words(data):
    data["nb_words"] = data["text"].apply(lambda x: len(x.split(" ")))
    return data


def extract_lda(data, lda_model, cv):
    corpus = cv.transform(data["text_clean"])
    lda_output = lda_model.transform(corpus)
    lda_df = pd.DataFrame(data=lda_output, index=data.index, columns=[i for i in range(0, lda_model.n_components)])
    lda_df.columns = ["topic_" + str(x) for x in lda_df.columns]
    data = pd.concat([data, lda_df], axis=1)
    return data


def extract_doc2vec(data, model):
    doc2vec_df = data["text_clean"].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
    doc2vec_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df.columns]
    doc2vec_df.index = data.index
    data = pd.concat([data, doc2vec_df], axis=1)
    return data

def remove_columns_return_features(data):
    ignore_cols = ["text", "text_clean", "rating"]
    features = [c for c in data.columns if c not in ignore_cols]
    return data[features]


def extract_tfidf(data, tfidf):
    tfidf_result = tfidf.transform(data["text_clean"])
    tfidf_df = pd.DataFrame(data=tfidf_result.toarray(), columns=tfidf.get_feature_names_out())
    tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
    tfidf_df.index = data.index
    data = pd.concat([data, tfidf_df], axis=1)
    return data


def generate_features_sentiment(review, rating, lda_model, cv):
    data = pd.DataFrame([[review, rating]], columns=["text", "rating"])
    data = preprocess_text(data)
    data = extract_vader(data)
    data = extract_num_char(data)
    data = extract_num_words(data)
    data = extract_lda(data, lda_model, cv)
    features = remove_columns_return_features(data)
    return features


def generate_features_fakereview(review, rating, doc2vec_model, lda_model, tfidf, cv):
    data = pd.DataFrame([[review, rating]], columns=["text", "rating"])
    data = preprocess_text(data)
    data = extract_num_char(data)
    data = extract_num_words(data)
    data = extract_doc2vec(data, doc2vec_model)
    data = extract_tfidf(data, tfidf)
    data = extract_lda(data, lda_model, cv)
    features = remove_columns_return_features(data)
    return features


@app.route("/detection_server", methods=["POST"])
@cross_origin(origin="*", headers=["Content-Type", "Authorization"])
def main():
    data = request.get_json()
    review = data.get("description", "")
    rating = data.get("userScore", "")

    start = time.time()
    features_sentiment = generate_features_sentiment(
        review,
        rating,
        model_lda_sentiment,
        model_cv_sentiment
    )
    features_fakereview = generate_features_fakereview(
        review,
        rating,
        model_doc2vec_fakereview,
        model_lda_fakereview,
        model_tfidf_fakereview,
        model_cv_fakereview
    )
    end = time.time()
    print("FEATURE GENERATION TIME: ", (end-start) * 10**3, "ms")

    start = time.time()
    prediction_sentiment = model_clf_sentiment.predict(features_sentiment)
    prediction_fakereview = model_clf_fakereview.predict(features_fakereview)
    end = time.time()
    print("PREDICTION TIME: ", (end-start) * 10**3, "ms")

    if prediction_fakereview == 1 and prediction_sentiment == 1:
        return jsonify({"fakeResult": False, "sentimentResult": False})
    elif prediction_fakereview == 1 and prediction_sentiment == 0:
        return jsonify({"fakeResult": False, "sentimentResult": True})
    elif prediction_fakereview == 0 and prediction_sentiment == 1:
        return jsonify({"fakeResult": True, "sentimentResult": False})
    elif prediction_fakereview == 0 and prediction_sentiment == 0:
        return jsonify({"fakeResult": True, "sentimentResult": True})


if __name__ == "__main__":
    app.run(port=5000, debug=True)
