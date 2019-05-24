# Citation: Used code from in-class demo to help create our query expansion json
# for each word.
# link to the in class demo: http://www.cs.cornell.edu/courses/cs4300/2019sp/Demos/demo20_2.html


import json
import pickle
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds

with open('./data/word_id_lookup.json') as wil_file:
    word_to_index = json.load(wil_file)

with open('./data/inverted_dict_id_word.json') as wil_file:
    index_to_word = json.load(wil_file)

data = pickle.load(open( './data/tfidf.pickle', "rb" ) )
my_matrix = data.transpose()

from scipy.sparse.linalg import svds
u, s, v_trans = svds(my_matrix, k=100)

words_compressed, _, docs_compressed = svds(my_matrix, k=40)
docs_compressed = docs_compressed.transpose()

from sklearn.preprocessing import normalize
words_compressed = normalize(words_compressed, axis = 1)

def closest_words(word_in, k = 10):
    if word_in not in word_to_index: return "Not in vocab."
    sims = words_compressed.dot(words_compressed[word_to_index[word_in],:])
    asort = np.argsort(-sims)[:k+1]
    return [(index_to_word[str(i)],sims[i]/sims[asort[0]]) for i in asort[1:]]

dict = {}

for key in word_to_index:
    dict[key] = [x[0] for x in closest_words(key)]

with open('./data/query_expansion.json', "w") as f:
    json.dump(dict, f)
