import sys
import string
import re
import numpy as np
import pandas as pd
import numba as nb

from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import NearMiss
from sklearn.svm import SVC

translator = str.maketrans("", "", string.punctuation)
stop_words = stopwords.words("english")
stop_words = stop_words + ["subject", "com", "are", "edu", "would", "could"]
lemmatizer = WordNetLemmatizer()
scaler = StandardScaler()

def file_read(filename):
    with open(filename, "r") as file:
        data = file.readlines()
    return data

def pre_process(data):
    word_freq = Counter()

    for i, sentence in enumerate(data):
        sentence = sentence.lower() # LOWER CASE
        sentence = re.sub(r"\d+", "", sentence) # NUMBERS
        sentence = sentence.translate(translator) # PUNCTUATION
        sentence = " ".join(sentence.split()) # WHITESPACE
        sentence = [lemmatizer.lemmatize(word) for word in word_tokenize(sentence) if word not in stop_words] # LEMMATIZATION, STOPWORDS
        data[i] = sentence

        for word in sentence: # WORD FREQUENCY
            word_freq[word] += 1

    return data, word_freq

def generate_vocab(word_freq, threshold=0):
    vocabulary, vocabulary_id2word = {}, {}

    for j, word in enumerate(word_freq):
        if word_freq[word] >= threshold:
            vocabulary[word] = j
            vocabulary_id2word[j] = word

    return vocabulary, vocabulary_id2word

def generate_corpus(data, vocabulary):
    corpus = []

    for sentence in data:
        document = [word for word in sentence if word in vocabulary]
        corpus_d = []

        for word in document:
            corpus_d.append(vocabulary[word])
        corpus.append(np.asarray(corpus_d))

    return corpus

@nb.njit(nogil=True, parallel=True)
def gibbs_sampling(corpus, num_iter=200):
    topic_assignment = []
    for doc in corpus:
        topic_assignment.append(np.random.randint(low=0, high=NUMBER_OF_TOPICS, size=len(doc)))

    # THETA
    topic_dist_over_doc = np.zeros((NUMBER_OF_DOCUMENTS, NUMBER_OF_TOPICS))
    for d in range(NUMBER_OF_DOCUMENTS):
        for t in range(NUMBER_OF_TOPICS):
            topic_dist_over_doc[d, t] = np.sum(topic_assignment[d] == t)

    # PHI
    word_dist_over_topic = np.zeros((NUMBER_OF_TOPICS, VOCAB_SIZE))
    for i, doc in enumerate(corpus):
        for j, word in enumerate(doc):
            topic = topic_assignment[i][j]
            word_dist_over_topic[topic, word] += 1

    doc_topic_count = np.sum(topic_dist_over_doc, axis=1)
    topic_word_count = np.sum(word_dist_over_topic, axis=1)
    #doc_topic_count = np.zeros((NUMBER_OF_DOCUMENTS))

    for _ in nb.prange(num_iter):
        for d, doc in enumerate(corpus):
            for w, word in enumerate(doc):
                topic = topic_assignment[d][w]

                topic_dist_over_doc[d, topic] -= 1
                word_dist_over_topic[topic, word] -= 1
                topic_word_count[topic] -= 1

                p_d_t = (topic_dist_over_doc[d] + ALPHA) / (doc_topic_count[d] - 1 + NUMBER_OF_TOPICS * ALPHA)
                p_t_w = (word_dist_over_topic[:, w] + BETA) / (topic_word_count + VOCAB_SIZE * BETA)
                p_z = p_d_t * p_t_w
                p_z /= np.sum(p_z)
                #p_z = np.abs(topic_dist_over_doc[d, :] + ALPHA) * (word_dist_over_topic[:, word] + BETA) / (topic_word_count[:] + BETA * VOCAB_SIZE)
                new_topic = np.random.multinomial(1, p_z).argmax()

                topic_assignment[d][w] = new_topic

                topic_dist_over_doc[d, new_topic] += 1
                word_dist_over_topic[new_topic, word] += 1
                topic_word_count[new_topic] += 1

    return topic_assignment, topic_dist_over_doc, word_dist_over_topic, doc_topic_count, topic_word_count

np.random.seed(42)
data = file_read(sys.argv[1])
data, word_freq = pre_process(data)
vocabulary, vocabulary_id2word = generate_vocab(word_freq)
corpus = generate_corpus(data, vocabulary)

NUMBER_OF_DOCUMENTS = len(corpus)
VOCAB_SIZE = len(vocabulary)
NUMBER_OF_TOPICS = 15
ALPHA = 0.1
BETA = 0.1

z_assignment, ndk, nkw, nd, nk = gibbs_sampling(corpus, num_iter=20000)

metadata = pd.read_csv("dataset/metadata.txt", header=None, sep="\\s+")
metadata.columns = ["date", "rid", "uid", "pid", "label", "a", "b", "c", "rate"]

feature_vectors = ndk / nd.reshape(NUMBER_OF_DOCUMENTS, 1)
feature_vectors = scaler.fit_transform(feature_vectors)
labels = np.array(metadata["label"].values)
labels = labels == "N"

x_train, x_test, y_train, y_test = train_test_split(feature_vectors, labels, test_size=0.3, random_state=42)

pipe = Pipeline([
    ("sampler", NearMiss(sampling_strategy=0.66666666666)),
    ("clf", SVC(random_state=42))
])

grid = GridSearchCV(
    pipe,
    param_grid = {
        'clf__C': [0.1, 1, 10, 100, 1000],
        'clf__gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        'clf__kernel': ['rbf']
    },
    scoring = "f1",
    n_jobs = -1,
    verbose = 3
)

grid.fit(x_train, y_train)

print(grid.best_params_)
print(grid.best_score_)

y_pred = grid.predict(x_test)

print(classification_report_imbalanced(y_test, y_pred))
print("ROC-AUC: ", roc_auc_score(y_test, y_pred))
