from operator import le
import matplotlib
matplotlib.use('Agg')
import logging as log
import logging as log
import sys
import nltk
# For Python 3.6 we use the base keras
from tensorflow import keras
#from tensorflow import keras
import sys
import numpy as np
import random
from difflib import SequenceMatcher
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.tokenize.treebank import TreebankWordDetokenizer
from config import VOCAB_SIZE, DIVERSITY_METRIC
import Levenshtein as lev
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import re

INDEX_FROM=3   # word index offset


def untokenize(vector):
    return TreebankWordDetokenizer().detokenize(vector)    

def find_adjs(text):
    tokenized_text = word_tokenize(text)
    word_tags = nltk.pos_tag(tokenized_text)
    adjs_advs = [i for i in range(0,len(word_tags)) if word_tags[i][1] in ['JJ', 'JJR', 'JJS']]
    return word_tags, adjs_advs


def get_synonym(word):
    word = word.lower()
    synonyms = []
    synsets = wordnet.synsets(word)
    if (len(synsets) == 0):
        return []
    for synset in synsets:
        lemma_names = synset.lemma_names()
        for lemma_name in lemma_names:
            lemma_name = lemma_name.lower().replace('_', ' ')
            if (lemma_name != word and lemma_name not in synonyms):
                synonyms.append(lemma_name)
    if len(synonyms) > 0:
        sword = random.choice(synonyms)
        return sword
    else:
        return None


def listToString(s): 
    
    # initialize an empty string
    str1 = s[0] 
    
    # traverse in the string  
    for ele in s[1:]: 
        if isinstance(ele, str):
            str1 += "." + ele  
    
    # return string  
    return str1 

def decode_imdb_reviews(embd):

    word_to_id = keras.datasets.imdb.get_word_index()
    word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2
    word_to_id["<UNUSED>"] = 3

    id_to_word = {value:key for key,value in word_to_id.items()}
    text = ' '.join(id_to_word[id] for id in embd)

    return text



def compute_sparseness(map, x):
    n = len(map)
    # Sparseness is evaluated only if the archive is not empty
    # Otherwise the sparseness is 1
    if (n == 0) or (n == 1):
        sparseness = 0
    else:
        sparseness = density(map, x)
    return sparseness

def get_neighbors(b):
    neighbors = []
    neighbors.append((b[0], b[1]+1))
    neighbors.append((b[0]+1, b[1]+1))
    neighbors.append((b[0]-1, b[1]+1))
    neighbors.append((b[0]+1, b[1]))
    neighbors.append((b[0]+1, b[1]-1))
    neighbors.append((b[0]-1, b[1]))
    neighbors.append((b[0]-1, b[1]-1))
    neighbors.append((b[0], b[1]-1))

    return neighbors

def density(map, x):
    b = x.features
    density = 0
    count = 0
    neighbors = get_neighbors(b)
    for neighbor in neighbors:
        if neighbor not in map:
            density += 1
    return density


def get_distance(ind1, ind2):
    """ Computes distance based on configuration """

    if DIVERSITY_METRIC == "INPUT":
        # input space
        distance = levenshtein(ind1.text, ind2.text)
    
    elif DIVERSITY_METRIC == "LATENT":
        # latent space
        distance = euclidean(ind1.latent_vector, ind2.latent_vector)

    elif DIVERSITY_METRIC == "HEATMAP":
        # heatmap space
        distance = euclidean(ind1.explanation, ind2.explanation)
    

    return distance

def get_distance_by_metric(ind1, ind2, metric):
    """ Computes distance based on metric """

    if metric == "INPUT":
        # input space
        distance = levenshtein(ind1.text, ind2.text)
    
    elif metric == "LATENT":
        # latent space
        distance = euclidean(ind1.latent_vector, ind2.latent_vector)

    elif metric == "HEATMAP":
        # heatmap space
        distance = euclidean(ind1.explanation, ind2.explanation)
    

    return distance


def euclidean(img1, img2):
    dist = np.linalg.norm(img1 - img2)
    return dist

def manhattan(coords_ind1, coords_ind2):
    return abs(coords_ind1[0] - coords_ind2[0]) + abs(coords_ind1[1] - coords_ind2[1])


def levenshtein(v1, v2):
    return lev.distance(v1, v2)
    # return SequenceMatcher(None, v1, v2).ratio()


def rescale_map(features, perfs, new_min_1, new_max_1, new_min_2, new_max_2):
    if new_max_1 > 25:
        shape_1 = 25
    else:
        shape_1 = new_max_1 + 1
    
    if new_max_2 > 25:
        shape_2 = 25
    else:
        shape_2 = new_max_2 + 1

    output = dict()

    original_bins1 = np.linspace(new_min_1, new_max_1, shape_1)
    original_bins2 = np.linspace(new_min_2, new_max_2, shape_2)

    for key, value in perfs.items():
        i = key[0]
        j = key[1]
        if i < new_max_1 and j < new_max_2:
            new_i = np.digitize(i, original_bins1, right=False)
            new_j = np.digitize(j, original_bins2, right=False)
            if (new_i, new_j) not in output or value < output[(new_i, new_j)]:
                output[(new_i, new_j)] = value
    return output

# Useful function that shapes the input in the format accepted by the ML model.


def setup_logging(log_to, debug):

    def log_exception(extype, value, trace):
        log.exception('Uncaught exception:', exc_info=(extype, value, trace))

    # Disable annoyng messages from matplot lib.
    # See: https://stackoverflow.com/questions/56618739/matplotlib-throws-warning-message-because-of-findfont-python
    log.getLogger('matplotlib.font_manager').disabled = True

    term_handler = log.StreamHandler()
    log_handlers = [term_handler]
    start_msg = "Started test generation"

    if log_to is not None:
        file_handler = log.FileHandler(log_to, 'a', 'utf-8')
        log_handlers.append( file_handler )
        start_msg += " ".join(["writing to file: ", str(log_to)])

    log_level = log.DEBUG if debug else log.INFO

    log.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=log_level, handlers=log_handlers)

    sys.excepthook = log_exception

    log.info(start_msg)

def compute_area_under_curve(x, y):
    area = np.trapz(y=y, x=x)
    return area






nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('vader_lexicon')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')

lemma = WordNetLemmatizer()
stop_words = stopwords.words('english')


file = open('opinion-lexicon-English/negative-words.txt', 'r')
neg_words = file.read().split()

file = open('opinion-lexicon-English/positive-words.txt', 'r')
pos_words = file.read().split()

def text_prep(x):
    corp = str(x).lower()
    corp = re.sub('[^a-zA-Z]+', ' ', corp).strip()
    tokens = word_tokenize(corp)
    words = [t for t in tokens if t not in stop_words]
    lemmatize = [lemma.lemmatize(w) for w in words]
    return lemmatize


def compute_sentiment(text):
    preprocess_text = text_prep(text)
    total_len = len(text)
    num_pos = len([i for i in preprocess_text if i in pos_words])
    num_neg = len([i for i in preprocess_text if i in neg_words])

    sentiment = round((num_pos - num_neg)/total_len , 2)
    # sentiment = round(num_pos/(num_neg+1),2)

    return sentiment

def count_pos(text):
    
    preprocess_text = text_prep(text)
    num_pos = len([i for i in preprocess_text if i in pos_words])
    return num_pos

def count_neg(text):
    preprocess_text = text_prep(text)
    num_neg = len([i for i in preprocess_text if i in neg_words])
    return num_neg

def count_pos_relative(text):
    preprocess_text = text_prep(text)

    
    num_pos = len([i for i in preprocess_text if i in pos_words])
    total = count_words(text)

    if total > 0:
        return int((num_pos/total)*100)
    else:
        return 0

def count_neg_relative(text):
    preprocess_text = text_prep(text)
    num_neg = len([i for i in preprocess_text if i in neg_words])
    total = count_words(text)
    if total > 0:
        return int((num_neg/total)*100)
    else:
        return 0



def compute_sentiment2(text):
    sent = SentimentIntensityAnalyzer()
    polarity = [round(sent.polarity_scores(i)['compound'], 2) for i in text]
    return polarity



def count_words(text):
    txt = text.split()
    count = 0
    for word in txt:
        if word not in [".", ",", ":", ";", "<", ">", "/", "!", "?", "br"]:
            count += 1
    return count

    
def count_words_using_vec(text):
    from sklearn.feature_extraction.text import CountVectorizer
    # create the transform
    vectorizer = CountVectorizer()
    # tokenize and build vocab
    vectorizer.fit(text)
    # summarize
    print(vectorizer.vocabulary_)
    # encode document
    vector = vectorizer.transform(text)
    # summarize encoded vector
    print(vector.shape)
    print(type(vector))
    print(vector.toarray())
    return sum(vector)


def count_verbs(text):
    tokenized_text = word_tokenize(text)
    word_tags = nltk.pos_tag(tokenized_text)
    verbs = [wt[0] for wt in word_tags if wt[1] in ['VB', 'VBD', 'VBN', 'VBP', 'VBZ', 'VBG']]
    return len(verbs)


def count_adjs(text):
    tokenized_text = word_tokenize(text)
    word_tags = nltk.pos_tag(tokenized_text)
    adjs_advs = [wt[0] for wt in word_tags if wt[1] in ['JJ', 'JJR', 'JJS']]
    return len(adjs_advs)



def feature_simulator(function, x):
    """
    Calculates the value of the desired feature
    :param function: name of the method to compute the feature value
    :param x: genotype of candidate solution x
    :return: feature value
    """
    if function == 'count_neg':
        return count_neg(x.text)
    if function == 'count_pos':
        return count_pos(x.text)
    if function == 'count_words':
        return count_words(x.text)
    if function == 'rel_count_pos':
        return count_pos_relative(x.text)
    if function == 'rel_count_neg':
        return count_neg_relative(x.text)
    if function == 'count_verbs':
        return count_verbs(x.text)
    if function == 'count_adjs':
        return count_adjs(x.text)