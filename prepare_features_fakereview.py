import os
import joblib
import spacy
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.preprocessing import StandardScaler
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from textblob import TextBlob

tqdm.pandas()
nlp = spacy.load("en_core_web_sm")

if not os.path.exists("./models"):
    os.makedirs("./models")


def read_file(filename: str) -> pd.DataFrame:
    data = pd.DataFrame()
    if filename == "dataset/yelpCHI_hotel_text.txt" or filename == "dataset/yelpCHI_restaurant_text.txt":
        with open(filename, "r") as f:
            data = f.readlines()
            data = pd.DataFrame(data, columns=["text"]).apply(pd.Series)
    elif filename == "dataset/yelpCHI_hotel_meta.txt" or filename == "dataset/yelpCHI_restaurant_meta.txt":
        data = pd.read_csv(filename, header=None, sep=' ', names=["date", "rid", "uid", "pid", "label", "o1", "o2", "o3", "rating"])
        data = data[["label"]]
    return data


def generate_labels(data: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    print("Generating Labels...")
    data["is_fake"] = metadata["label"].progress_apply(lambda x: 1 if x == "Y" or x == -1 else 0)
    return data


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


def clean_text(texts):
    clean_texts = []
    for tokens in nlp.pipe(tqdm(texts), n_process=-1):
        tokens = [
            token.lemma_.lower()
            for token in tokens
            if token.is_alpha
            and not token.is_stop
            and check_pos_tag(token)
        ]
        tokens = [t for t in tokens if len(t) > 1]
        text = " ".join(tokens)
        clean_texts.append(text)
    return clean_texts


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    print("Preprocessing Text...")
    data.insert(len(data.columns), "text_clean", clean_text(data["text"]))
    return data


def feature_extract_vader_sentiment(data):
    print("Generating Sentiment Features...")
    data["sentiments"] = data["text"].progress_apply(SentimentIntensityAnalyzer().polarity_scores)
    data = pd.concat([data.drop(["sentiments"], axis=1), data["sentiments"].progress_apply(pd.Series)], axis=1)
    data["subjectivity"] = data["text"].progress_apply(
        lambda x:
            TextBlob(x).sentiment.subjectivity
    )
    return data


def feature_extract_num_char(data):
    print("Generating NumCharacter Features...")
    data["nb_chars"] = data["text"].progress_apply(lambda x: len(x))
    return data


def feature_extract_num_words(data):
    print("Generating NumWords Features...")
    data["nb_words"] = data["text"].progress_apply(lambda x: len(x.split(" ")))
    return data


def feature_extract_tfidf(data, min_df=1, max_df=1.0, max_features=None):
    print("Generating TFIDF features...")
    tfidf = TfidfVectorizer(
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        ngram_range=(1, 2),
        dtype=np.float32
    )
    tfidf_result = tfidf.fit_transform(data["text_clean"])
    tfidf_result = StandardScaler(with_mean=False).fit_transform(tfidf_result)
    tfidf_df = pd.DataFrame(
        data=tfidf_result.toarray(),
        columns=tfidf.get_feature_names_out()
    )
    tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
    tfidf_df.index = data.index
    data = pd.concat([data, tfidf_df], axis=1)
    return data, tfidf


def feature_extract_doc2vec(data, vector_size=300):
    print("Generating Tagged Document...")
    documents = [
        TaggedDocument(doc, [i])
        for i, doc in enumerate(
            data["text_clean"].progress_apply(lambda x: x.split(" "))
        )
    ]
    model = Doc2Vec(
        documents=documents,
        vector_size=vector_size,
        dm_mean=1,
        workers=-1
    )
    print("Generating Doc2Vec Features...")
    doc2vec_df = data["text_clean"].progress_apply(
        lambda x:
            model.infer_vector(x.split(" "))
    ).apply(pd.Series)
    doc2vec_df = pd.DataFrame(StandardScaler().fit_transform(doc2vec_df))
    doc2vec_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df.columns]
    doc2vec_df.index = data.index
    data = pd.concat([data, doc2vec_df], axis=1)
    return data, model


def feature_extract_lda(data, num_topics):
    print("Generating LDA features...")
    cv = TfidfVectorizer(
        min_df=2,
        max_df=0.95,
        max_features=2500,
        ngram_range=(1, 2),
    )
    corpus = cv.fit_transform(data["text_clean"])
    print("Corpus shape: ", corpus.shape)
    lda_model = LDA(
        n_components=num_topics,
        learning_method="online",
        doc_topic_prior=0.1,
        topic_word_prior=0.01,
        max_doc_update_iter=1000,
        total_samples=corpus.shape[0],
        batch_size=int(corpus.shape[0] / 250),
        n_jobs=-1,
        random_state=0
    )
    lda_model = lda_model.partial_fit(corpus)
    print("LDA Score: ", lda_model.score(corpus))
    print("LDA Perplexity: ", lda_model.perplexity(corpus))
    lda_output = lda_model.transform(corpus)
    lda_output = StandardScaler().fit_transform(lda_output)
    lda_df = pd.DataFrame(
        data=lda_output,
        index=data.index,
        columns=[i for i in range(0, lda_model.n_components)]
    )
    lda_df.columns = ["topic_" + str(x) for x in lda_df.columns]
    data = pd.concat([data, lda_df], axis=1)
    return data, lda_model, cv


def remove_text_columns(data):
    label = "is_fake"
    ignore_cols = [label, "text", "text_clean"]
    features = [c for c in data.columns if c not in ignore_cols]
    return data[features], data[label]


def main():
    hotel_metadata = read_file("dataset/yelpCHI_hotel_meta.txt")
    hotel_data = read_file("dataset/yelpCHI_hotel_text.txt")
    restaurant_metadata = read_file("dataset/yelpCHI_restaurant_meta.txt")
    restaurant_data = read_file("dataset/yelpCHI_restaurant_text.txt")
    data = pd.concat([hotel_data, restaurant_data], axis=0)
    metadata = pd.concat([hotel_metadata, restaurant_metadata], axis=0)
    data = generate_labels(data, metadata)
    data = preprocess_data(data)
    data = feature_extract_num_char(data)
    data = feature_extract_num_words(data)
    data = feature_extract_vader_sentiment(data)
    # data, doc2vec_model = feature_extract_doc2vec(data, vector_size=1000)
    # joblib.dump(doc2vec_model, "models/fakereview_model_doc2vec.sav")
    data, tfidf = feature_extract_tfidf(data, min_df=2, max_df=0.95, max_features=2500)
    joblib.dump(tfidf, "models/fakereview_model_tfidf.sav")
    data, lda_model, cv = feature_extract_lda(data, num_topics=15)
    joblib.dump(lda_model, "models/fakereview_model_lda.sav")
    joblib.dump(cv, "models/fakereview_model_cv.sav")
    features, target = remove_text_columns(data)
    joblib.dump(features, "models/fakereview_features.sav")
    joblib.dump(target, "models/fakereview_target.sav")
    print(features)


if __name__ == "__main__":
    main()
