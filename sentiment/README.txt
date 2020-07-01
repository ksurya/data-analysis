Sentiment of product reviews on Amazon, IMDB and Yelp
========================================

Motivations:
===========

- How operations on word vectors influence the performance of sentence classification tasks?
- Can we tease out context from contextual embeddings vectors such as BERT/ELMo?

TODO:
====


Experiment
=======

- Performance on learning multiple embeddings is analyzed
  - output: results/results.csv, viz: results/results.jmp
  - Experiment Names:
    - FrozenGlove: fixed glove representations (best acc: 82%)
    - Glove: Glove repr. finetuned (best acc: 82%)
    - Embedding: Representations initialized with xavier uniform (best acc: 61%)
    - FrozenGlove + Embedding: two vectors added. addition of Glove in theory shouldn't change loss. (best acc: 79%)
    - FrozenGlove * Embedding: Frozen Glove and Embedding multiplied (best acc: 76%)
    - FrozenGlove ; Embedding: Frozen Glove and Embedding concatenated (best acc: 80%)
    - Glove + Embedding: best acc 72%
    - Glove * Embedding: best acc 54%
    - Glove ; Embedding: best acc 82%
  - Best acc of all experiments aren't always at same step. That's an interesting takeaway. See JMP results

- Model
  - 1 layer GRU, 25d Glove word vectors
  - Binary cross entropy loss, Adam optimizer, 1e-3 lr, 32 batch size, 120 epochs


