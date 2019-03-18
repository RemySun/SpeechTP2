import numpy as np
import nltk

from copy import deepcopy

def normalize_counts(counts):
    """ Normalize the counts so that we have a probability conditioned on the left hand side
    """
    new_counts = deepcopy(counts)
    for key in counts:
        left_occurences = sum(counts[key].values())

        for right_side in counts[key]:
            new_counts[key][right_side] /= left_occurences

    return new_counts



def extract_pcfg_tree(tree, pcfg, pl):
    """ Extract rules with counts from an annotated tree IN PLACE
    """
    label = tree.label().split('-')[0] # Forget POS details

    if label not in pcfg:
        pcfg[label] = {}
    if label not in pl:
        pl[label] = {}

    # Extract representation of the right side of the rule
    right_side = []
    for child in tree:

        if type(child) == nltk.Tree: # Recursively extract rules lower in the tree
            child_label = child.label().split('-')[0]

            right_side.append(child_label)

            extract_pcfg_tree(child,pcfg, pl)

        else: # Separately extract POS to word rules
            child_label = child
            if child_label not in pl[label]:
                pl[label][child_label] = 1
            else:
                pl[label][child_label] += 1

    if right_side:
        if tuple(right_side) not in pcfg[label]:
            pcfg[label][tuple(right_side)] = 1
        else:
            pcfg[label][tuple(right_side)] += 1



def clean_counts(counts):
    """ Eliminate grammar entry for left side that do not occur
    """
    new_counts = deepcopy(counts)
    for key in counts:
        if not counts[key]:
            del new_counts[key]

    return new_counts



def get_n_aries(cnf,n=1):
    """ Extract rules with right hand side of length n
    """
    n_aries = {left_side:{right_side: right_sides[right_side] for right_side in right_sides if len(right_side) == n} for left_side,right_sides in cnf.items()}

    n_aries = clean_counts(n_aries)

    return n_aries
