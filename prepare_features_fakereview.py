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
    if filename == "dataset/textdata.txt":
        with open(filename, "r") as f:
            data = f.readlines()
            data = pd.DataFrame(data, columns=["text"])
    elif filename == "dataset/metadata.txt":
        data = pd.read_csv(filename, header=None, sep=" ", names=["date", "rid", "uid", "pid", "label", "o1", "o2", "o3", "rating"])
    return data


def generate_labels(data: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    print("Generating Labels...")
    data["is_fake"] = metadata["label"].progress_apply(lambda x: 1 if x == "Y" else 0)
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
    data["text_clean_tokens"] = data["text_clean"].progress_apply(lambda x: x.split(" "))
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


def remove_allzerorows(smatrix):
    nonzero_row_indice, _ = smatrix.nonzero()
    unique_nonzero_indice = np.unique(nonzero_row_indice)
    return smatrix[unique_nonzero_indice]


def feature_extract_lda(data):
    print("Generating LDA features...")
    cv = CountVectorizer(min_df=2, max_df=0.95, ngram_range=(1, 2))
    corpus = cv.fit_transform(data["text_clean"])
    corpus = remove_allzerorows(corpus)
    lda_model = LDA(
        doc_topic_prior=0.1,
        topic_word_prior=0.01,
        n_components=10,
        n_jobs=-1,
        random_state=0
    ).fit(corpus)
    lda_output = lda_model.transform(corpus)
    lda_df = pd.DataFrame(data=lda_output, columns=lda_model.get_feature_names_out())
    lda_df.columns = ["topic_" + str(x) for x in lda_df.columns]
    lda_df.index = data.index
    data = pd.concat([data, lda_df], axis=1)
    return data, lda_model


def remove_text_columns(data):
    label = "is_fake"
    ignore_cols = [label, "text", "text_clean", "text_clean_tokens"]
    features = [c for c in data.columns if c not in ignore_cols]
    return data[features], data[label]


def main():
    data = read_file("dataset/textdata.txt")
    metadata = read_file("dataset/metadata.txt")
    data = generate_labels(data, metadata)
    data = preprocess_data(data)
    data = feature_extract_vader_sentiment(data)
    data = feature_extract_num_char(data)
    data = feature_extract_num_words(data)
    data, lda_model = feature_extract_lda(data)
    joblib.dump(lda_model, "models/fakereview_model_lda.sav")
    # data, tfidf = feature_extract_tfidf(data, min_df=2, max_df=0.5, max_features=1000)
    # joblib.dump(tfidf, "models/fakereview_model_tfidf.sav")
    features, target = remove_text_columns(data)
    joblib.dump(features, "models/fakereview_features.sav")
    joblib.dump(target, "models/fakereview_target.sav")
    print(features)


if __name__ == "__main__":
    main()
