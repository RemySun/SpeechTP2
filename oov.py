import numpy as np
import pickle
from helper_functions.levenshtein import *
from scipy.spatial.distance import cosine

path_to_data = './data/'

polyglot = pickle.load(open(path_to_data + 'polyglot-fr.pkl','rb'),encoding='latin1')

polyglot = dict(zip(polyglot[0],polyglot[1]))

lexicon = pickle.load(open(path_to_data + 'lexicon.p','rb'))
known_words = list(lexicon.keys())

split='dev'

corpus = []
with open(path_to_data + 'raw_{}.txt'.format(split),'r') as raw_file:
    for sentence in raw_file:
        corpus.append(sentence[:-1].split())

oov_forms = {}

lev_mult=0.5

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


with open(path_to_data + 'oov_{}'.format(split),'wb+') as oov_file:
    pickle.dump(oov_forms,oov_file)
