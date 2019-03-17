###
### This file contains helper functions pertaining to the use of the CYK algorithm
###

import nltk
import numpy as np
from copy import deepcopy

######################################
### Functions used for CYK parsing ###
######################################


def unroll(preds,current_cell,variables_symb,terminals_symb):
    """ Recursively unrolls the predecessor table given by CYK
    """
    if current_cell[0] == 0:
        level, position, variable = current_cell
        return nltk.Tree(variables_symb[current_cell[2]],[terminals_symb[preds[level,position,variable][0]]])

    else:
        level, position, variable = current_cell
        new_level, var1, var2 = preds[level,position,variable]
        return nltk.Tree(variables_symb[variable],
                         [unroll(preds,(new_level,position,var1),variables_symb,terminals_symb),
                          unroll(preds,(level-new_level-1,position+new_level+1,var2),variables_symb,terminals_symb)])



def cyk(sentence,unaries,binaries,inv_pl,pl):
    """ Parse a sentence with the CYK algorithm, returns the parse tree given grammar cnf and grammar lexicon
    """

    # Define some mappings for convenience
    variables_symb = dict(enumerate(list(binaries.keys()) + list(unaries.keys())))
    variables_idx = {symb:idx for (idx,symb) in variables_symb.items()}

    terminals_symb = dict(enumerate(pl))
    terminals_idx = {symb:idx for (idx,symb) in terminals_symb.items()}

    n = len(sentence)
    r = len(variables_symb)

    # Collect the possible POS tags with associated probabilities P(word|tag)
    POS_tags = []
    for word in sentence:
        POS_tags.append(inv_pl[word])


    probs = np.zeros((n,n,r)) + np.inf
    preds = np.zeros((n,n,r,3),dtype = np.int32)

    # Fill out the first row of the cyk charts using the unary terminal rules of the cnf
    for i in range(n):
        for variable in range(r):
            if variables_symb[variable] in unaries:
                right_sides = unaries[variables_symb[variable]]
                for tag in POS_tags[i]:
                    if (tag,) in right_sides:
                        probs[0,i,variable] = - np.log(right_sides[(tag,)]) - np.log(POS_tags[i][tag])
                        preds[0,i,variable] = (terminals_idx[tag],-1,-1)


    # Iteratively fill out the rows of the charts using the binary rules of the cnf
    for l in range(1,n):
        for s in range(n-l):
            for p in range(l):
                for variable in range(r):
                    if variables_symb[variable] in binaries:
                        right_sides = binaries[variables_symb[variable]]
                        for right_side in right_sides:
                            var1,var2 = right_side
                            if probs[p,s,variables_idx[var1]] < np.inf and probs[l-p-1,s+p+1,variables_idx[var2]] < np.inf:
                                prob_splitting = - np.log(right_sides[(var1,var2)]) + probs[p,s,variables_idx[var1]] + probs[l-p-1,s+p+1,variables_idx[var2]]
                                if probs[l,s,variable] > prob_splitting:
                                    probs[l,s,variable] = prob_splitting
                                    preds[l,s,variable] = [p,variables_idx[var1],variables_idx[var2]]

    return unroll(preds,(n-1,0,0),variables_symb,terminals_symb)



#########################################
### Functions used for CNF conversion ###
#########################################


def unit_rounding(pcfg,terminals):
    """Perform one round of the UNIT procedure to necessary to convert to a CNF grammar
    """
    unit_rules = []
    pcfg = deepcopy(pcfg)

    for left_side in pcfg:
        for right_side in pcfg[left_side]:
            if len(right_side) == 1 and right_side[0] not in terminals:
                unit_rules.append([left_side, right_side[0]])

    ############## DEAL WITH UNIT RULES IN UNIT RULES ?
    for unit_rule in unit_rules:
        for left_side in pcfg:
            for right_side in pcfg[left_side]:
                if unit_rule[1]==left_side and unit_rule[0]!=left_side:
                    pcfg[unit_rule[0]][right_side] = pcfg[unit_rule[0]][(unit_rule[1],)]*pcfg[left_side][right_side]

    for unit_rule in unit_rules:
        del pcfg[unit_rule[0]][(unit_rule[1],)]

    return pcfg



def convert_cnf(pcfg,terminals):
    """Convert a PCFG to a CNF grammar
    """
    pcfg = deepcopy(pcfg)

    new_symb_count = 0


    ### Eliminating rules with non terminals mixed in
    new_pcfg = deepcopy(pcfg)
    for left_side in pcfg:
        right_sides = pcfg[left_side]
        for right_side in right_sides:
            new_rs = []
            hit_term = False
            if len(right_side) > 1:
                for symb in right_side:
                    if symb in terminals:
                        print()
                        new_pcfg['TERM'+str(new_symb_count)] = {(symb,):1} #### CHECK COMPUTATION
                        new_rs.append('TERM'+str(new_symb_count))
                        new_symb_count += 1
                        hit_term = True
                    else:
                        new_rs.append(symb)
                new_pcfg[left_side][tuple(new_rs)] = pcfg[left_side][right_side] #### CHECK COMPUTATION
                if hit_term:
                    del new_pcfg[left_side][right_side]

    pcfg = new_pcfg


    ### Eliminating rules with more than 1 non terminal
    new_pcfg = deepcopy(pcfg)
    for left_side in pcfg:
        right_sides = pcfg[left_side]
        for right_side in right_sides:
            new_rs = []
            if len(right_side) > 2:
                unfolding_symbol = left_side
                for symb in right_side[:-2]:
                    if unfolding_symbol == left_side:
                        new_pcfg[unfolding_symbol][(symb,'BIN'+str(new_symb_count))] = pcfg[left_side][right_side] #### CHECK COMPUTATION needs to retake value for the first
                    else:
                        new_pcfg[unfolding_symbol] = {(symb,'BIN'+str(new_symb_count)):1} #### CHECK COMPUTATION needs to retake value for the first
                    unfolding_symbol = 'BIN'+str(new_symb_count)
                    new_symb_count += 1
                new_pcfg[unfolding_symbol] = {right_side[-2:]:1} #### CHECK COMPUTATION
                del new_pcfg[left_side][right_side]

    pcfg = new_pcfg

    ### Eliminating useless rules
    unit_loops = 0
    tmp = {}
    while unit_loops < 1000 and tmp != pcfg:
        tmp = pcfg
        pcfg= unit_rounding(pcfg,terminals)
        unit_loops += 1

    return pcfg
