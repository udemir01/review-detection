import os
import joblib
import spacy
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.preprocessing import StandardScaler

tqdm.pandas()
nlp = spacy.load("en_core_web_sm")

if not os.path.exists("./models"):
    os.makedirs("./models")


def generate_labels(data):
    print("Generating Labels...")
    data["text"] = data["Negative_Review"] + data["Positive_Review"]
    data["is_negative"] = data["Reviewer_Score"].progress_apply(lambda x: 1 if x < 5 else 0)
    data = data[["text", "is_negative"]]
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


def remove_no_neg_pos(data):
    print("Replacing No Negative/Positive...")
    data["text"] = data["text"].progress_apply(lambda x: x.replace("No Negative", "").replace("No Positive", ""))
    return data


def preprocess_data(data):
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


def feature_extract_doc2vec(data, vector_size=200):
    print("Generating Tagged Document...")
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(data["text_clean"].progress_apply(lambda x: x.split(" ")))]
    model = Doc2Vec(documents, vector_size=vector_size, workers=-1)
    print("Generating Doc2Vec Features...")
    doc2vec_df = data["text_clean"].progress_apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
    doc2vec_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df.columns]
    doc2vec_df.index = data.index
    data = pd.concat([data, doc2vec_df], axis=1)
    return data, model


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


def feature_extract_lda(data, num_topics=10):
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
    label = "is_negative"
    ignore_cols = [label, "text", "text_clean", "rating"]
    features = [c for c in data.columns if c not in ignore_cols]
    return data[features], data[label]


def main(sample_size=0.1):
    data = pd.read_csv("dataset/Hotel_Reviews.csv")
    data = data.sample(frac=sample_size, replace=False, random_state=0)
    data = generate_labels(data)
    data = remove_no_neg_pos(data)
    data = preprocess_data(data)
    data = feature_extract_vader_sentiment(data)
    # data = feature_extract_num_char(data)
    # data = feature_extract_num_words(data)
    data, lda_model, cv = feature_extract_lda(data, num_topics=15)
    joblib.dump(lda_model, "models/sentiment_model_lda.sav")
    joblib.dump(cv, "models/sentiment_model_cv.sav")
    # data, doc2vec_model = feature_extract_doc2vec(data, vector_size=1000)
    # joblib.dump(doc2vec_model, "models/sentiment_model_tfidf.sav")
    data, tfidf = feature_extract_tfidf(data, min_df=2, max_df=0.95, max_features=2500)
    joblib.dump(tfidf, "models/sentiment_model_tfidf.sav")
    features, target = remove_text_columns(data)
    joblib.dump(features, "models/sentiment_features.sav")
    joblib.dump(target, "models/sentiment_target.sav")

    print(features)


if __name__ == "__main__":
    main(sample_size=0.1)
