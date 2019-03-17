import nltk
import numpy as np
import pickle

from nltk.corpus import BracketParseCorpusReader
from copy import deepcopy
from helper_functions.parsing import *
from helper_functions.cyk import *

path_to_data = './data/'



#### Read the files and separate datasets ####

reader = BracketParseCorpusReader(path_to_data,r'.*\.mrg_strict')


corpus = [sent for sent in reader.parsed_sents()]
corpus_idx = [i for i in range(len(corpus))]

np.random.seed(0)
np.random.shuffle(corpus_idx)

idx_train = corpus_idx[:int(len(corpus)*0.8)]
idx_dev = corpus_idx[int(len(corpus)*0.8):int(len(corpus)*0.9)]
idx_test = corpus_idx[int(len(corpus)*0.9):]

corpus_train = [corpus[i] for i in idx_train]
corpus_dev = [corpus[i] for i in idx_dev]
corpus_test = [corpus[i] for i in idx_test]



#### Extract counts from tree, normalize and clean up symbols not on left hand ####

pcfg_counts = {}
pl_counts = {}
lexicon_counts = {}

for sent in corpus_train:
    extract_pcfg_tree(sent,pcfg_counts, pl_counts)

for tag in pl_counts: # also create an inverted lexicon for OOV
    for right_side in pl_counts[tag]:
        if right_side not in lexicon_counts.keys():
            lexicon_counts[right_side] = {}
        lexicon_counts[right_side][tag] = pl_counts[tag][right_side]



pcfg_counts = normalize_counts(pcfg_counts)
pl_counts = normalize_counts(pl_counts)
lexicon_counts = normalize_counts(lexicon_counts)

def clean_counts(counts):
    new_counts = deepcopy(counts)
    for key in counts.keys():
        if not counts[key]:
            del new_counts[key]
    return new_counts


pcfg = clean_counts(pcfg_counts)
pl = clean_counts(pl_counts)
lexicon = clean_counts(lexicon_counts)



#### Convert pcfg to cnf ####

terminals = [terminal for terminal in pl]

cnf = convert_cnf(pcfg, terminals)



#### Write data to file ####


## Write counts

with open(path_to_data + 'pcfg_grammar.p','wb+') as pcfg_file:
    pickle.dump(pcfg,pcfg_file)

with open(path_to_data + 'cnf_grammar.p','wb+') as pcfg_file:
    pickle.dump(cnf,pcfg_file)

with open(path_to_data + 'pcfg_lexicon.p','wb+') as pl_file:
    pickle.dump(pl,pl_file)

with open(path_to_data + 'lexicon.p','wb+') as lexicon_file:
    pickle.dump(lexicon,lexicon_file)


## Write raw inputs and labels

with open(path_to_data + 'raw_train.txt','w+') as raw_file:
    for sentence in corpus_train:
        raw_file.write(' '.join(sentence.leaves())+'\n')

with open(path_to_data + 'raw_dev.txt','w+') as raw_file:
    for sentence in corpus_dev:
        raw_file.write(' '.join(sentence.leaves())+'\n')

with open(path_to_data + 'raw_test.txt','w+') as raw_file:
    for sentence in corpus_test:
        raw_file.write(' '.join(sentence.leaves())+'\n')


with open(path_to_data + 'pos_train.txt','w+') as raw_file:
    for sentence in corpus_train:
        tags = [node[1] for node in sentence.pos()]
        raw_file.write(' '.join(tags)+'\n')

with open(path_to_data + 'pos_dev.txt','w+') as raw_file:
    for sentence in corpus_dev:
        tags = [node[1] for node in sentence.pos()]
        raw_file.write(' '.join(tags)+'\n')

with open(path_to_data + 'pos_test.txt','w+') as raw_file:
    for sentence in corpus_test:
        tags = [node[1] for node in sentence.pos()]
        raw_file.write(' '.join(tags)+'\n')
