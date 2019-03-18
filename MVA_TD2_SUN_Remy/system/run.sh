#!/bin/sh

while IFS= read -r line; do
    python3 oov.py $@ --dynamic_input "$line"
    python3 cyk.py $@ --dynamic_input "$line"
done

