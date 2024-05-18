import os
import joblib
import spacy
import pandas as pd
from tqdm.auto import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

tqdm.pandas()

if not os.path.exists("./models"):
    os.makedirs("./models")


def clean_text(texts):
    clean_texts = []
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    for _, tokens in enumerate(nlp.pipe(tqdm(texts), n_process=-1)):
        tokens = [
            token.lemma_.lower()
            for token in tokens
            if token.is_alpha
            and not token.is_stop
            and not token.is_space
        ]
        tokens = [t for t in tokens if len(t) > 1]
        text = " ".join(tokens)
        clean_texts.append(text)
    return clean_texts


def main(sample_size=0.1):
    data = pd.read_csv("dataset/Hotel_Reviews.csv")
    data["review"] = data["Negative_Review"] + data["Positive_Review"]
    print("Generating Labels...")
    data["is_bad_review"] = data["Reviewer_Score"].progress_apply(lambda x: 1 if x < 5 else 0)
    data = data[["review", "is_bad_review"]]
    data = data.sample(frac=sample_size, replace=False, random_state=42)
    print("Replacing No Negative/Positive...")
    data["review"] = data["review"].progress_apply(lambda x: x.replace("No Negative", "").replace("No Positive", ""))
    print("Preprocessing Text...")
    data.insert(len(data.columns), "review_clean", clean_text(data["review"]))
    print("Generating Sentiment Features...")
    data["sentiments"] = data["review"].progress_apply(SentimentIntensityAnalyzer().polarity_scores)
    data = pd.concat([data.drop(["sentiments"], axis=1), data["sentiments"].progress_apply(pd.Series)], axis=1)
    print("Generating NumCharacter Features...")
    data["nb_chars"] = data["review"].progress_apply(lambda x: len(x))
    print("Generating NumWords Features...")
    data["nb_words"] = data["review"].progress_apply(lambda x: len(x.split(" ")))
    print("Generating Tagged Document...")
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(data["review_clean"].progress_apply(lambda x: x.split(" ")))]
    model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=-1)
    joblib.dump(model, "models/sentiment_model_doc2vec.sav")
    print("Saved Doc2Vec Model.")
    print("Generating Doc2Vec Features...")
    doc2vec_df = data["review_clean"].progress_apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
    doc2vec_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df.columns]
    data = pd.concat([data, doc2vec_df], axis=1)
    tfidf = TfidfVectorizer(max_df=10, max_features=500)
    tfidf_result = tfidf.fit_transform(data["review_clean"]).toarray()
    joblib.dump(tfidf, "models/sentiment_model_tfidf.sav")
    print("Saved TFIDF Model.")
    tfidf_df = pd.DataFrame(tfidf_result, columns=tfidf.get_feature_names_out())
    tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
    tfidf_df.index = data.index
    data = pd.concat([data, tfidf_df], axis=1)
    label = "is_bad_review"
    ignore_cols = [label, "review", "review_clean"]
    features = [c for c in data.columns if c not in ignore_cols]
    x, y = SMOTE(sampling_strategy="minority", n_jobs=-1).fit_resample(data[features], data[label])
    joblib.dump(x, "models/sentiment_features.sav")
    joblib.dump(y, "models/sentiment_labels.sav")
    print("Saved Features and Labels for Accuracy Testing.")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(x, y)
    joblib.dump(clf, "models/sentiment_model_clf.sav")
    print("Saved ML Model.")


main()
