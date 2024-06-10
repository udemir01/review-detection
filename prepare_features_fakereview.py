import os
import joblib
import spacy
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

tqdm.pandas()

if not os.path.exists("./models"):
    os.makedirs("./models")


def read_file(filename: str) -> pd.DataFrame:
    data = pd.DataFrame()
    if filename == "dataset/yelpCHI_text.txt":
        with open(filename, "r") as f:
            data = f.readlines()
            data = pd.DataFrame(data, columns=["text"])
    elif filename == "dataset/yelpCHI_meta.txt":
        data = pd.read_csv(filename, header=None, sep=' ', names=["date", "rid", "uid", "pid", "label", "o1", "o2", "o3", "rating"])
        data = data[["label"]]
    elif filename == "dataset/yelpNYC_text.txt":
        data = pd.read_csv(filename, header=None, sep='\t', names=["uid", "pid", "date", "text"])
        data = data[["text"]]
    elif filename == "dataset/yelpNYC_meta.txt":
        data = pd.read_csv(filename, header=None, sep='\t', names=["uid", "pid", "rating", "label", "date"])
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
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    for tokens in nlp.pipe(tqdm(texts), n_process=6):
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
    tfidf_df = pd.DataFrame(data=tfidf_result.toarray(), columns=tfidf.get_feature_names_out())
    tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
    tfidf_df.index = data.index
    data = pd.concat([data, tfidf_df], axis=1)
    return data, tfidf


def feature_extract_doc2vec(data, vector_size=300):
    print("Generating Tagged Document...")
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(data["text_clean"].progress_apply(lambda x: x.split(" ")))]
    model = Doc2Vec(
        documents=documents,
        dm_mean=1,
        min_count=1,
        window=10,
        alpha=0.065,
        min_alpha=0.065,
        vector_size=vector_size,
        workers=-1
    )
    print("Generating Doc2Vec Features...")
    doc2vec_df = data["text_clean"].progress_apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
    doc2vec_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df.columns]
    doc2vec_df.index = data.index
    data = pd.concat([data, doc2vec_df], axis=1)
    return data, model


def feature_extract_lda(data, num_topics):
    print("Generating LDA features...")
    cv = CountVectorizer(ngram_range=(1, 2))
    corpus = cv.fit_transform(data["text_clean"])
    lda_model = LDA(
        n_components=num_topics,
        doc_topic_prior=0.1,
        topic_word_prior=0.01,
        max_iter=1000,
        learning_method="online",
        n_jobs=6,
        random_state=0
    )
    lda_output = lda_model.fit_transform(corpus)
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
    metadata = read_file("dataset/yelpCHI_meta.txt")
    data = read_file("dataset/yelpCHI_text.txt")
    data = generate_labels(data, metadata)
    data = preprocess_data(data)
    # data, doc2vec_model = feature_extract_doc2vec(data, vector_size=1000)
    # joblib.dump(doc2vec_model, "models/fakereview_model_doc2vec.sav")
    # data, tfidf = feature_extract_tfidf(data, min_df=10, max_df=0.95, max_features=10000)
    # joblib.dump(tfidf, "models/fakereview_model_tfidf.sav")
    data, lda_model, cv = feature_extract_lda(data, 15)
    joblib.dump(lda_model, "models/fakereview_model_lda.sav")
    joblib.dump(cv, "models/fakereview_model_cv.sav")
    features, target = remove_text_columns(data)
    joblib.dump(features, "models/fakereview_features.sav")
    joblib.dump(target, "models/fakereview_target.sav")
    print(features)


if __name__ == "__main__":
    main()
