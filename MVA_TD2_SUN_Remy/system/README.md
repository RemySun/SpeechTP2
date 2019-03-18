# TP2: Parsing. Rémy Sun

The data subfolder contains trained grammars, lexicons and other necessary pieces.

The script run.sh reads from stdin to output parsed tree (from a different grammar than the one in the original treebank).

The command cat test_text.txt | ./run.sh was tested to work properly for some dummy test_text.txt with proper content.

This code requires the libraries pickle, multiprocessins, numpy, scipy, sklearn and nltk to run.

## Arguments

'--lev_mult' default=0.5 help='Weight of spelling distance'

'--n_cores' default=8 help='Number of cpu cores to parallelize on'

Do **not** use the --dynamic_input option, it is already used by the shell script.

There are a few other options but they are not used in dynamic mode, which the script executes in. They can be found at the beginning of cyk.py
