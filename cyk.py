import numpy as np
import pickle
import time
import nltk

from copy import deepcopy
from multiprocessing import Pool

from helper_functions.cyk import *
from helper_functions.parsing import *
import argparse

parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('--split', default='dev',
                    help='Data split to use')

parser.add_argument('--lev_mult', default=0.5,type=int,
                    help='Weight of spelling distance')
parser.add_argument('--n_cores', default=22,type=int,
                    help='Number of cpu cores to parallelize on')

parser.add_argument('--evaluate', default=True, type = bool,
                    help='Do we need to evaluate')


parser.add_argument('--path_to_data', default='./data/',type=str,
                    help='Where is the data ?')
parser.add_argument('--input_path', default=None,type=str,
                    help='Where is the input file ?')
parser.add_argument('--pos_path', default=None,type=str,
                    help='Where is the label file ?')

parser.add_argument('--dynamic_input', default=None,type=str,
                    help='If defined, feeds a sentence to the system.')


args = parser.parse_args()

print(args)

split = args.split
lev_mult = args.lev_mult
evaluate = args.evaluate
input_path = args.input_path
pos_path = args.pos_path
path_to_data = args.path_to_data
n_cores = args.n_cores
dynamic_input = args.dynamic_input

#### Reading files that were already processed ####

if dynamic_input != None:
    corpus = [dynamic_input.split()]
    evaluate = False
else:
    corpus = []
    if not input_path:
        input_path = path_to_data + 'raw_{}.txt'.format(split)
    with open(input_path,'r') as raw_file:
        for sentence in raw_file:
            corpus.append(sentence[:-1].split())

if evaluate:
    labels = []
    if not pos_path:
        pos_path = path_to_data + 'pos_{}.txt'.format(split)
    with open(path_to_data + 'pos_{}.txt'.format(split),'r') as raw_file:
        for sentence in raw_file:
            labels.append([tag.split('-')[0] for tag in sentence[:-1].split()])

with open(path_to_data + 'cnf_grammar.p','rb') as cnf_file:
    cnf = pickle.load(cnf_file)

with open(path_to_data + 'pcfg_lexicon.p','rb') as pl_file:
    pl = pickle.load(pl_file)

with open(path_to_data + 'lexicon.p','rb') as lexicon_file:
    lexicon = pickle.load(lexicon_file)


with open(path_to_data + 'oov_{}_{}.p'.format(split,lev_mult),'rb') as oov_file:
    oov = pickle.load(oov_file)
    
for word in oov:
    pl[oov[word]][word] = 1

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
    return cyk(sentence,unaries,binaries,inv_pl,pl,cnf)

p=Pool(n_cores)

parsed_trees = p.map(evaluate_sentence,corpus)

print(time.time() - start)

#### Evaluate ####

if evaluate:
    problems = []
    OK = []
    
    count = 0
    match = 0
    minimum_count = 0
    minimum_match = 0

    for i in range(len(parsed_trees)):
        POS_tags = parsed_trees[i].leaves()
        for j in range(len(POS_tags)):
            count += 1
            if corpus[i][j] in lexicon:
                minimum_count += 1
            if POS_tags[j] == labels[i][j]:
                match +=1
                if corpus[i][j] in lexicon:
                    minimum_match += 1
                
print("We got an accuracy of {} !".format(match/count))
print("We got an accuracy of {} ! On stuff that was in the lexicon !".format(minimum_match/minimum_count))
