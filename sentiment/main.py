from torch.nn.utils.rnn import pad_sequence
from types import SimpleNamespace

import os
import requests
import spacy
import torch
import pandas as pd
import numpy as np
import torch.utils.data as D
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def create_tokenizer():
    nlp = spacy.load("en")

    def caller(sent):
        return [tok.text.lower() for tok in nlp(sent)]

    return caller


def create_token_indexer(vocab, update_vocab=True):

    def caller(word):
        if update_vocab:
            vocab.add(word)
        index = vocab.get_index(word)
        return vocab.UNK if index is None else index

    return caller


class Vocabulary(object):

    def __init__(self):
        self.token2index = {}
        self.index2token = {}
        self.PAD = self.add("PAD")
        self.UNK = self.add("UNK")
        self.SOS = self.add("SOS")
        self.EOS = self.add("EOS")

    def __len__(self):
        return len(self.index2token)

    def add(self, word):
        if word not in self.token2index:
            index = len(self.token2index)
            self.token2index[word] = index
            self.index2token[index] = word
            return index

    def get_word(self, index):
        return self.index2token.get(index)

    def get_index(self, token):
        return self.token2index.get(token)


def train_test_split(dataset, train_ratio):
    length = len(dataset)
    train_len = int(length * train_ratio)
    test_len = length - train_len
    return D.random_split(dataset, [train_len, test_len])


def get_glove_vectors(words):
    resp = requests.get("http://localhost:8989", params={"words": ",".join(words)})
    return resp.json()["vectors"]


def get_embedding_weights(vocab):
    if os.path.isfile("data/embedding.pkl"):
        print("Loading weights from cache")
        return torch.load("data/embedding.pkl")
    embed_weights = []
    special_words = []
    for i, w in enumerate(special_words):
        vec = [0] * 25
        vec[i] = 1
        special_words[w] = vec
    for idx in range(len(vocab)):
        word = vocab.get_word(idx)
        try:
            weight = get_glove_vectors([word])[0]
        except:
            weight = np.random.randn(25)
            special_words.append(word)
        embed_weights.append(weight)
    print(special_words)
    embed_weights = np.array(embed_weights)
    torch.save(embed_weights, "data/embedding.pkl")
    return embed_weights


class Reviews(D.Dataset):

    def __init__(self, fname, tokenizer, indexer):
        self.instances = {
            "review": [],
            "review_tokens": [],
            "review_vectors": [],
            "score": [],
        }
        self.df = pd.read_csv(fname, sep="\t", header=None, names=["review", "score"])
        for i, row in self.df.iterrows():
            tokens = ["SOS"] + tokenizer(row["review"]) + ["EOS"]
            indices = [indexer(i) for i in tokens]
            self.instances["review"].append(row["review"])
            self.instances["review_tokens"].append(tokens)
            self.instances["review_vectors"].append(indices)
            self.instances["score"].append(row["score"])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        return {col: self.instances[col][idx] for col in self.instances}


def create_mini_batcher(padding=0):

    def mini_batcher(samples):
        batch = {k: [] for k in samples[0]}
        for sample in samples:
            batch["review"].append(sample["review"])
            batch["review_tokens"].append(sample["review_tokens"])
            batch["review_vectors"].append(torch.tensor(sample["review_vectors"]))
            batch["score"].append(sample["score"])
        batch["review_vectors"] = pad_sequence(batch["review_vectors"], batch_first=True, padding_value=padding)
        batch["score"] = torch.tensor(batch["score"])
        return batch

    return mini_batcher


def calc_accuracy(pred_logits, target_labels):
    pred_labels = (pred_logits > 0.5)
    acc = (pred_labels.float() == target_labels.float()).float().sum()
    return acc / pred_labels.shape[0]


class Classifier(nn.Module):

    def __init__(self, embed_weights):
        super().__init__()
        embed_size = 25
        self.emb = nn.Embedding.from_pretrained(embed_weights, freeze=True)
        self.l0 = nn.GRU(embed_size, embed_size, batch_first=True)
        self.l1 = nn.Linear(embed_size, 1)
        self.criterion = nn.BCEWithLogitsLoss()

        for name, param in self.named_parameters():
            if name.startswith("l0"):
                if "weights" in name:
                    nn.init.xavier_uniform_(param)
                else:
                    param.data.fill_(0) 

    def forward(self, batch):
        # embedding
        x = self.emb(batch["review_vectors"])
        
        # rnn 
        x, h = self.l0(x)
        h = h[-1]

        # final FC layer 
        y = self.l1(h).squeeze(-1)

        # calc loss and acc
        loss = self.criterion(y, batch["score"].float())
        acc = calc_accuracy(torch.sigmoid(y), batch["score"])
        return {"score": x, "loss": loss, "acc": acc}


def metrics_store():

    store = {}
    counter = {}

    def add(**kw):
        for k, v in kw.items():
            store[k] = store.get(k, 0) + v
            counter[k] = counter.get(k, 0) + 1
    
    def formatted(prefix=""):
        kw = {prefix + k: v/counter[k] for k, v in store.items()}
        return ", ".join("{}: {:.4f}".format(k, v) for k, v in kw.items())

    return SimpleNamespace(add=add, formatted=formatted)


def train_and_test(train_ds, test_ds, mini_batcher, batch_size, lr, epochs, model):
    model = move_to_device(model, 0)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_iterator = D.DataLoader(train_ds, batch_size=batch_size, collate_fn=mini_batcher)
    test_iterator = D.DataLoader(test_ds, batch_size=batch_size, collate_fn=mini_batcher)
    for ep in range(epochs):
        tr_metrics = metrics_store()
        te_metrics = metrics_store()
        for batch in train_iterator:
            batch = move_to_device(batch, 0)
            optimizer.zero_grad()
            output = model(batch)
            output["loss"].backward()
            optimizer.step()
            tr_metrics.add(loss=output["loss"].item(), acc=output["acc"].item())
        with torch.no_grad():
            for batch in test_iterator:
                batch = move_to_device(batch, 0)
                output = model(batch)
                te_metrics.add(loss=output["loss"].item(), acc=output["acc"].item())
        
        if ep % 10 == 0:
            print("Epoch: {:4d}, {}, {}".format(ep, tr_metrics.formatted("train_"), te_metrics.formatted("test_")))


def move_to_device(obj, device):
    if type(obj) is list:
        return [move_to_device(o, device) for o in obj]
    elif type(obj) is dict:
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif type(obj) is torch.Tensor or isinstance(obj, nn.Module):
        return obj.to(device)
    return obj


def main():
    print("building vocab")
    vocab = Vocabulary()
    tokenizer = create_tokenizer()
    indexer = create_token_indexer(vocab, update_vocab=True)
    mini_batcher = create_mini_batcher(padding=vocab.PAD)
    
    ds1 = Reviews("data/amazon_cells_labelled.txt", tokenizer, indexer)
    ds2 = Reviews("data/imdb_labelled.txt", tokenizer, indexer)
    ds3 = Reviews("data/yelp_labelled.txt", tokenizer, indexer)
    ds = D.ConcatDataset([ds1, ds2, ds3])
    train_ds, test_ds = train_test_split(ds, 0.8)

    print("getting glove weights")
    embed_weights = torch.tensor(get_embedding_weights(vocab)).float().to(0)

    print("loading model")
    model = Classifier(
        embed_weights=embed_weights,
    )

    print("starting training..")
    train_and_test(
        train_ds=train_ds,
        test_ds=test_ds,
        mini_batcher=mini_batcher,
        batch_size=32,
        lr=0.001,
        epochs=1000, 
        model=model, 
    )


if __name__ == "__main__":
    main()
