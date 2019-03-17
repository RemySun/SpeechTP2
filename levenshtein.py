import numpy as np


def levenshtein(s1,s2):
    n1 = len(s1)
    n2 = len(s2)

    partial_distances = np.zeros((n1+1,n2+1),dtype=np.int32)

    # Initialize the partial distances matrix
    partial_distances[:,0] = [i for i in range(n1+1)]
    partial_distances[0,:] = [i for i in range(n2+1)]

    for i in range(n1):
        for j in range(n2):
            lateral_changes = min(partial_distances[i,j+1]+1,partial_distances[i+1,j]+1)

            if s1[i] == s2[j]:
                partial_distances[i+1,j+1] = min(lateral_changes, partial_distances[i,j])
            else:
                partial_distances[i+1,j+1] = min(lateral_changes, partial_distances[i,j] + 1)

    return partial_distances[-1,-1]
