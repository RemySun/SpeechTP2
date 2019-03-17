import levenshtein
import pytest

from helper_functions.cyk import *

def test_levenshtein():
    assert levenshtein.levenshtein('test','tilt') == 2
    assert levenshtein.levenshtein('a','') == 1
    assert levenshtein.levenshtein('a','a') == 0
    assert levenshtein.levenshtein('a','abx') == 2

def test_cnf():
    test_pcfg = {'NP':{('ADJ', 'N'): 0.6,('N',):0.4}, 'N':{('cat',):0.2,('dog',):0.8}}
    test_cnf = {'NP':{('ADJ', 'N'): 0.6,('cat',):0.08,('dog',):0.32}, 'N':{('cat',):0.2,('dog',):0.8}}
    test_terminals = ['cat','dog']

    assert  {
        left_side:
        {
            right_side: pytest.approx(right_sides[right_side]) for right_side in right_sides
        } for left_side,right_sides in convert_cnf(test_pcfg,test_terminals).items()
    } == test_cnf
