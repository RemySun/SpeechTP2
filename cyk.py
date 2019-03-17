import numpy as np
import pickle
import time
import numba
from numba import jit
import nltk

from copy import deepcopy
from multiprocessing import Pool

from helper_functions.cyk import *
from helper_functions.parsing import *

split = 'train'
path_to_data = './data/'


#### Reading files that were already processed ####

corpus = []
with open(path_to_data + 'raw_{}.txt'.format(split),'r') as raw_file:
    for sentence in raw_file:
        corpus.append(sentence[:-1].split())

labels = []
with open(path_to_data + 'pos_{}.txt'.format(split),'r') as raw_file:
    for sentence in raw_file:
        labels.append([tag.split('-')[0] for tag in sentence[:-1].split()])

with open(path_to_data + 'cnf_grammar.p','rb') as cnf_file:
    cnf = pickle.load(cnf_file)

with open(path_to_data + 'pcfg_lexicon.p','rb') as pl_file:
    pl = pickle.load(pl_file)



#### Doing some last minute processing on the data ####

inv_pl = {}
for tag in pl:
    for word in pl[tag]:
        if word in inv_pl:
            inv_pl[word][tag] = pl[tag][word]
        else:
            inv_pl[word] = {tag:pl[tag][word]}

unaries = get_n_aries(cnf,n=1)
binaries = get_n_aries(cnf,n=2)



#### Parse the sentences in parallel ####

start = time.time()
def evaluate_sentence(sentence):
    return cyk(sentence,unaries,binaries,inv_pl,pl)

p=Pool(8)

parsed_trees = p.map(evaluate_sentence,corpus[:10])

print(time.time() - start)



#### Evaluate ####

count = 0
match = 0
for i in range(len(parsed_trees)):
    POS_tags = parsed_trees[i].leaves()
    for j in range(len(POS_tags)):
        count += 1
        if POS_tags[j] == labels[i][j]:
            match +=1

print("We got an accuracy of {} !".format(match/count))
