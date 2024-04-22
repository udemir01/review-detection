import sys
import numpy as np
import pandas as pd
import numba as nb
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer

def generate_docs_vocab_from_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    data = pd.DataFrame(lines)[0].sample(frac=1.0, random_state=42).values

    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tf = tf_vectorizer.fit_transform(data)
    vocabulary = tf_vectorizer.vocabulary_

    docs = []
    for row in tf.toarray():
        present_words = np.where(row != 0)[0].tolist()
        present_words_with_count = []
        for word_idx in present_words:
            for _ in range(row[word_idx]):
                present_words_with_count.append(word_idx)
        docs.append(present_words_with_count)

    return docs, vocabulary

def gibbs_sampling(docs):
    z_d_n = [[0 for _ in range(len(d))] for d in docs]  # z_i_j
    theta_d_z = np.zeros((NUMBER_OF_DOCS, NUMBER_OF_TOPICS))
    phi_z_w = np.zeros((NUMBER_OF_TOPICS, NUMBER_OF_VOCAB))
    n_d = np.zeros((NUMBER_OF_DOCS))
    n_z = np.zeros((NUMBER_OF_TOPICS))

    ## Initialize the parameters
    # m: doc id
    for d, doc in enumerate(docs):
        # n: id of word inside document, w: id of the word globally
        for n, w in enumerate(doc):
            # assign a topic randomly to words
            z_d_n[d][n] = n % NUMBER_OF_TOPICS
            # get the topic for word n in document m
            z = z_d_n[d][n]
            # keep track of our counts
            theta_d_z[d][z] += 1
            phi_z_w[z, w] += 1
            n_z[z] += 1
            n_d[d] += 1

    for _ in range(200):
        for d, doc in enumerate(docs):
            for n, w in enumerate(doc):
                # get the topic for word n in document m
                z = z_d_n[d][n]

                # decrement counts for word w with associated topic z
                theta_d_z[d][z] -= 1
                phi_z_w[z, w] -= 1
                n_z[z] -= 1

                # sample new topic from a multinomial according to our formula
                p_d_t = (theta_d_z[d] + ALPHA) / (n_d[d] - 1 + NUMBER_OF_TOPICS * ALPHA)
                p_t_w = (phi_z_w[:, w] + BETA) / (n_z + NUMBER_OF_VOCAB * BETA)
                p_z = p_d_t * p_t_w
                p_z /= np.sum(p_z)
                new_z = np.random.multinomial(1, p_z).argmax()

                # set z as the new topic and increment counts
                z_d_n[d][n] = int(new_z)
                theta_d_z[d][new_z] += 1
                phi_z_w[new_z, w] += 1
                n_z[new_z] += 1

    return theta_d_z, phi_z_w


docs, vocabulary = generate_docs_vocab_from_file(sys.argv[1])

NUMBER_OF_DOCS = len(docs)
NUMBER_OF_VOCAB = len(vocabulary)
NUMBER_OF_TOPICS = 15

ALPHA = 1 / NUMBER_OF_TOPICS
BETA = 1 / NUMBER_OF_TOPICS

theta_d_z, phi_z_w = gibbs_sampling(docs)

inv_vocabulary = {v: k for k, v in vocabulary.items()}
n_top_words = 10
for topic_idx, topic in enumerate(phi_z_w):
    message = "Topic #%d: " % topic_idx
    message += " ".join([inv_vocabulary[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
    print(message)
