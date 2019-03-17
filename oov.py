import numpy as np
import pickle
from helper_functions.levenshtein import *
from scipy.spatial.distance import cosine
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

#print(args)

split = args.split
lev_mult = args.lev_mult
evaluate = args.evaluate
input_path = args.input_path
pos_path = args.pos_path
path_to_data = args.path_to_data
n_cores = args.n_cores
dynamic_input = args.dynamic_input

polyglot = pickle.load(open(path_to_data + 'polyglot-fr.pkl','rb'),encoding='latin1')

polyglot = dict(zip(polyglot[0],polyglot[1]))

lexicon = pickle.load(open(path_to_data + 'lexicon.p','rb'))
known_words = list(lexicon.keys())

if dynamic_input != None:
    corpus = [dynamic_input.split()]
else:
    if not input_path:
        input_path = path_to_data + 'raw_{}.txt'.format(split)
    with open(input_path,'r') as raw_file:
        for sentence in raw_file:
            corpus.append(sentence[:-1].split())

oov_forms = {}

i =0
for sentence in corpus:
    for token in sentence:
        if token not in lexicon:
            tags_scores = {}
            i+=1
            print(i)
            for word in known_words:
                dist_lev = np.exp(-bounded_levenshtein(word,token,k=3))

                dist_cos = 0
                if word in polyglot:
                    if token in polyglot:
                        dist_cos = 1- cosine(polyglot[word],polyglot[token])

                if dist_lev > 0 and dist_cos > 0:
                    for tag in lexicon[word]:
                        if tag in tags_scores:
                            tags_scores[tag] += (lev_mult * dist_lev + dist_cos) * lexicon[word][tag]
                        else:
                            tags_scores[tag] = (lev_mult * dist_lev + dist_cos) * lexicon[word][tag]
            if not tags_scores:
                # We default to a proper noun if we cannot find anything similar
                tag = 'NPP'
            else:
                tag_idx = np.argmax(list(tags_scores.values()))
                tag = list(tags_scores)[tag_idx]

            oov_forms[token] = tag


with open(path_to_data + 'oov_{}_{}.p'.format(split,lev_mult),'wb+') as oov_file:
    pickle.dump(oov_forms,oov_file)
