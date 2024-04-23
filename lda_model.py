import sys
import string
import re
import numba as nb
import numpy as np
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from numba_progress import ProgressBar

translator = str.maketrans("", "", string.punctuation)
stop_words = stopwords.words("english")
stop_words = stop_words + ["subject", "com", "are", "edu", "would", "could"]
lemmatizer = WordNetLemmatizer()

def file_read(filename):
    with open(filename, "r") as file:
        data = file.readlines()
    return data

def pre_process(data):
    word_freq = Counter()
    corpus = []

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

def generate_vocab(word_freq):
    vocabulary, vocabulary_id2word = {}, {}

    for j, word in enumerate(word_freq):
        if word_freq[word] >= 3:
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

@nb.njit(nogil=True)
def gibbs_sampling(corpus, progress, num_iter=200):
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

    for _ in range(num_iter):
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
                new_topic = np.random.multinomial(1, p_z).argmax()

                topic_assignment[d][w] = new_topic

                topic_dist_over_doc[d, new_topic] += 1
                word_dist_over_topic[new_topic, word] += 1
                topic_word_count[new_topic] += 1
        progress.update(1)

    return topic_assignment, topic_dist_over_doc, word_dist_over_topic, topic_word_count

data = file_read(sys.argv[1])
data, word_freq = pre_process(data)
vocabulary, vocabulary_id2word = generate_vocab(word_freq)
corpus = generate_corpus(data, vocabulary)

NUMBER_OF_DOCUMENTS = len(corpus)
VOCAB_SIZE = len(vocabulary)
NUMBER_OF_TOPICS = 20
NUMBER_OF_ITER = 200
ALPHA = 1 / NUMBER_OF_TOPICS
BETA = 1 / NUMBER_OF_TOPICS

with ProgressBar(total=NUMBER_OF_ITER) as progress:
    z_assignment, ndk, nkw, nk = gibbs_sampling(corpus, progress, NUMBER_OF_ITER)

inv_vocabulary = {v: k for k, v in vocabulary.items()}
n_top_words = 10
for topic_idx, topic in enumerate(nkw):
    message = "Topic #%d: " % topic_idx
    message += " ".join([inv_vocabulary[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
    print(message)
