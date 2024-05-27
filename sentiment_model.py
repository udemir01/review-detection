import os
import joblib
import spacy
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.preprocessing import StandardScaler

tqdm.pandas()

if not os.path.exists("./models"):
    os.makedirs("./models")


def generate_labels(data):
    print("Generating Labels...")
    data["review"] = data["Negative_Review"] + data["Positive_Review"]
    data["is_bad_review"] = data["Reviewer_Score"].progress_apply(lambda x: 1 if x < 5 else 0)
    data = data[["review", "is_bad_review"]]
    return data


def clean_text(texts):
    clean_texts = []
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    for tokens in nlp.pipe(tqdm(texts), n_process=-1):
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


def preprocess_data(data):
    print("Replacing No Negative/Positive...")
    data["review"] = data["review"].progress_apply(lambda x: x.replace("No Negative", "").replace("No Positive", ""))
    print("Preprocessing Text...")
    data.insert(len(data.columns), "review_clean", clean_text(data["review"]))
    return data


def feature_extract_vader_sentiment(data):
    print("Generating Sentiment Features...")
    data["sentiments"] = data["review"].progress_apply(SentimentIntensityAnalyzer().polarity_scores)
    data = pd.concat([data.drop(["sentiments"], axis=1), data["sentiments"].progress_apply(pd.Series)], axis=1)
    return data


def feature_extract_num_char(data):
    print("Generating NumCharacter Features...")
    data["nb_chars"] = data["review"].progress_apply(lambda x: len(x))
    return data


def feature_extract_num_words(data):
    print("Generating NumWords Features...")
    data["nb_words"] = data["review"].progress_apply(lambda x: len(x.split(" ")))
    return data


def feature_extract_doc2vec(data, vector_size=200):
    print("Generating Tagged Document...")
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(data["review_clean"].progress_apply(lambda x: x.split(" ")))]
    model = Doc2Vec(documents, vector_size=vector_size, window=2, min_count=1, workers=-1)
    print("Generating Doc2Vec Features...")
    doc2vec_df = data["review_clean"].progress_apply(lambda x: StandardScaler().fit_transform(model.infer_vector(x.split(" ")))).apply(pd.Series)
    doc2vec_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df.columns]
    data = pd.concat([data, doc2vec_df], axis=1)
    return data, model


def feature_extract_tfidf(data, min_df=1, max_df=1, max_features=None):
    tfidf = TfidfVectorizer(min_df=min_df, max_df=max_df, max_features=max_features)
    tfidf_result = tfidf.fit_transform(data["review_clean"]).toarray()
    tfidf_df = pd.DataFrame(tfidf_result, columns=tfidf.vocabulary_)
    tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
    tfidf_df.index = data.index
    data = pd.concat([data, tfidf_df], axis=1)
    return data, tfidf, tfidf_result


def feature_extract_lda(data, tfidf_result, num_topics=10):
    lda = LDA(n_components=num_topics, learning_offset=50, random_state=42, n_jobs=-1)
    W1 = lda.fit_transform(tfidf_result)
    colnames = ["topic_" + str(i) for i in range(lda.n_components)]
    df_doc_topic_pos = pd.DataFrame(np.round(W1,2),columns=colnames)
    significanttopic = np.argmax(df_doc_topic_pos.values,axis=1)
    df_doc_topic_pos["topic_dominance"] = significanttopic
    df_doc_topic_pos.columns = [str(x) for x in df_doc_topic_pos.columns]
    df_doc_topic_pos.index = data.index
    data = pd.concat([data, df_doc_topic_pos], axis=1)
    return data


def main(sample_size=0.1):
    data = pd.read_csv("dataset/Hotel_Reviews.csv")
    data = data.sample(frac=sample_size, replace=False, random_state=42)
    data = generate_labels(data)
    data = preprocess_data(data)
    data = feature_extract_vader_sentiment(data)
    data = feature_extract_num_char(data)
    data = feature_extract_num_words(data)
    data, model = feature_extract_doc2vec(data)
    joblib.dump(model, "models/sentiment_model_doc2vec.sav")
    data, tfidf, tfidf_result = feature_extract_tfidf(data, min_df=0.01, max_df=0.8, max_features=500)
    joblib.dump(tfidf, "models/sentiment_model_tfidf.sav")
    data = feature_extract_lda(data, tfidf_result, num_topics=25)
    joblib.dump(data, "models/sentiment_data.sav")

    print(data)


if __name__ == "__main__":
    main(sample_size=0.01)
