import sys
import tqdm
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim import corpora

translator = str.maketrans("", "", string.punctuation)
stop_words = stopwords.words("english")
stop_words = stop_words + ["subject", "com", "are", "edu", "would", "could"]
lemmatizer = WordNetLemmatizer()

def file_read(filename):
    with open(filename, "r") as file:
        doc = file.readlines()
    return doc

def pre_process(document):
    for i, sentence in enumerate(document):
        sentence = sentence.lower() # LOWER CASE
        sentence = re.sub(r"\d+", "", sentence) # NUMBERS
        sentence = sentence.translate(translator) # PUNCTUATION
        sentence = " ".join(sentence.split()) # WHITESPACE
        sentence = [lemmatizer.lemmatize(word) for word in word_tokenize(sentence) if word not in stop_words] # STOPWORDS, LEMMATIZATION
        document[i] = sentence
    return document

document = file_read(sys.argv[1])
document = pre_process(document)
vocabulary = corpora.Dictionary(document)
word_freq = [vocabulary.doc2bow(rev) for rev in document]

