import numpy as np
import pickle

path_to_data = './data/'

polyglot = pickle.load(open('polyglot-fr.pkl','rb'),encoding='latin1')

polyglot = dict(zip(polyglot[0],polyglot[1]))

lexicon = pickle.load(open('lexicon.p','rb'))

