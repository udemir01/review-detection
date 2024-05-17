import pandas as pd
import spacy
from tqdm.auto import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

tqdm.pandas()

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


def main():
    data = pd.read_csv("dataset/Hotel_Reviews.csv")
    data["review"] = data["Negative_Review"] + data["Positive_Review"]

    print("Sentiment Labeling")
    data["is_bad_review"] = data["Reviewer_Score"].progress_apply(lambda x: 1 if x < 5 else 0)
    data = data[["review", "is_bad_review"]]
    # data = data.sample(frac=0.5, replace=False, random_state=42)

    print("Replacing No Negative/Positive")
    data["review"] = data["review"].progress_apply(lambda x: x.replace("No Negative", "").replace("No Positive", ""))

    print("Text Preprocessing")
    data.insert(len(data.columns), "review_clean", clean_text(data["review"]))

    print("Vader Features")
    data["sentiments"] = data["review"].progress_apply(SentimentIntensityAnalyzer().polarity_scores)

    print("Turn Vader Features to Series")
    data = pd.concat([data.drop(["sentiments"], axis=1), data["sentiments"].progress_apply(pd.Series)], axis=1)

    print("Number of Characters Features")
    data["nb_chars"] = data["review"].progress_apply(lambda x: len(x))

    print("Number of Words Features")
    data["nb_words"] = data["review"].progress_apply(lambda x: len(x.split(" ")))

    print("TaggedDocument for Doc2Vec")
    documents = [
        TaggedDocument(doc, [i])
        for i, doc in enumerate(
            data["review_clean"].progress_apply(lambda x: x.split(" "))
        )
    ]

    model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=-1)
    print("Doc2Vec Model Created")

    print("Doc2Vec Features")
    doc2vec_df = data["review_clean"].progress_apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
    doc2vec_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df.columns]
    data = pd.concat([data, doc2vec_df], axis=1)

    tfidf = TfidfVectorizer(max_df=0.9, min_df=0.1, max_features=500)
    tfidf_result = tfidf.fit_transform(data["review_clean"]).toarray()
    tfidf_df = pd.DataFrame(
        tfidf_result,
        columns=tfidf.get_feature_names_out()
    )
    tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
    tfidf_df.index = data.index
    data = pd.concat([data, tfidf_df], axis=1)
    print("Tfidf Features Created")

    label = "is_bad_review"
    ignore_cols = [label, "review", "review_clean"]
    features = [c for c in data.columns if c not in ignore_cols]
    x_train, x_test, y_train, y_test = train_test_split(data[features], data[label], test_size=0.20, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    print(classification_report(y_test, y_pred))


main()
