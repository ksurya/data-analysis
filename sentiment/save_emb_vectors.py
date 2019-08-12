"""Save embedding weights to visualize in http://projector.tensorflow.org."""

import torch
from main import *


vocab = torch.load("models/vocab-01.pt")
model = torch.load("models/model-01.pt")

labels = []
glove = []
overlap = []

for index in range(len(vocab)):
    word = vocab.get_word(index)
    glove_vec = model.emb0.weight[index].tolist()
    overlap_vec = model.emb1.weight[index].tolist()

    labels.append("<{}>".format(word) + "\n")
    glove.append("\t".join([str(i) for i in glove_vec]) + "\n")
    overlap.append("\t".join([str(i) for i in overlap_vec]) + "\n")


with  \
    open("models/labels-01.csv", "w") as labels_f, \
    open("models/glove-01.csv", "w") as glove_f, \
    open("models/overlapped-01.csv", "w") as overlap_f:

    for l, g, o in zip(labels, glove, overlap):
        labels_f.write(l)
        glove_f.write(g)
        overlap_f.write(o)
