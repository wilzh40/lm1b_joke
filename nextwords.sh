#!/bin/bash

python ./lm_1b/lm1b_.py --mode next_words \
  --prefix_file "$1" \
  --pbtxt data/graph-2016-09-10.pbtxt \
  --vocab_file data/vocab-2016-09-10.txt \
  --ckpt 'data/ckpt-*' \
  --max_sample_words 15 --n_top_words 3
