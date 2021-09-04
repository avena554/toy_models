import os
from datasets import load_dataset, DatasetDict
from transformers import BertTokenizerFast, BertModel
import torch
from torch.utils.data import Dataset, DataLoader
from utils import Book, compose, write_book
import torch.nn as nn
from spacy.lang.en import English
from torch.nn.functional import pad
import re
import numpy as np

UNK = "<UNK>"
PAD = "<#>"
SAVE_PATH = "saved_models"
MODEL_NAME = "stupid"
SEED = 42
CUSTOM_E_DIM = 512
ID = nn.Identity()


truncate_corpus = False
corpus_size = 10
lr = 1e-3
b_size = 64

tag_re = re.compile("</?\\w+(?:\\s+\\w+=\"[^\\s\"]+\")*\\s*/?>")
extra_spaces = re.compile("\\s+")


# tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
# model = BertModel.from_pretrained("bert-base-cased")

def clean_extra_spaces(instance):
    return {"text": extra_spaces.sub(" ", instance['text'])}


def clean_html(instance):
    return {"text": tag_re.sub(" ", instance['text'])}


def lower(instance_as_dict):
    return {"text": instance_as_dict["text"].lower()}


def spacy_tokenizer():
    spacy_proc = English()
    return spacy_proc.tokenizer


def tokenize(instance, tokenizer):
    words = tokenizer(instance['text'])
    return {"words": [word.text for word in words], "text": instance['text']}


def add_words(instance, voc):
    for w in instance['words']:
        voc.add(w)

    return instance


def as_indices(sent, voc):
    t = torch.LongTensor([voc[w] for w in sent])
    return t


class TextIndexer:

    def __init__(self, voc):
        self._voc = voc

    def __call__(self, instance):
        instance.update({"x": as_indices(instance["words"], self._voc)})
        return instance


def as_tensor(instance):
    instance.update({'x': torch.LongTensor(instance["x"]), 'y': torch.FloatTensor([instance["label"]])})
    return instance


# Pytorch dataset to be fed to a pytorch DataLoader
class DecisionDataset(Dataset):

    def __init__(self, data_as_dict, prepare_fn):
        super(DecisionDataset, self).__init__()
        self._data = data_as_dict
        self._prepare_fn = prepare_fn

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._prepare_fn(self._data[index])


def gather_by_key(key, instances):
    return [instance[key] for instance in instances]


def pad_batch(instances, voc):
    max_size = max(instance['x'].size()[0] for instance in instances)
    paddings = [(0, max_size - instance['x'].size()[0]) for instance in instances]
    padded_instances = [pad(instance['x'], padding, mode="constant", value=voc[PAD])
                        for (instance, padding) in zip(instances, paddings)]
    instances_batch = torch.stack(padded_instances)
    labels_batch = torch.stack(gather_by_key('y', instances))
    masks_batch = instances_batch != voc[PAD]
    texts = gather_by_key("text", instances)
    words = gather_by_key("words", instances)

    return {'text': texts, 'words': words, 'x': instances_batch, 'y': labels_batch, 'mask': masks_batch}


class Perceptron(nn.Module):

    def __init__(self, in_s, nl=ID):
        super(Perceptron, self).__init__()
        self.in_s = in_s
        self.n_lin = nl
        self.lin = nn.Linear(in_s, 1)

    def forward(self, x):
        h = self.lin(x)
        out = self.n_lin(h)
        return out


class StupidSentenceEmbedding(nn.Module):

    def __init__(self):
        super(StupidSentenceEmbedding, self).__init__()

    def forward(self, w_es, masks):
        masked_w_es = w_es * masks.unsqueeze(2)
        s_lengths = torch.sum(masks, dim=1).unsqueeze(1)
        mean_e = torch.sum(masked_w_es, dim=1)/s_lengths
        return mean_e


class PerceptronOverCombinedWordEmbeddings(nn.Module):

    def __init__(self, embedding, composition, perceptron):
        super(PerceptronOverCombinedWordEmbeddings, self).__init__()
        self.perceptron = perceptron
        self.embedding = embedding
        self.composition = composition

    def forward(self, x, mask):
        e = self.embedding(x)
        s = self.composition(e, mask)
        return self.perceptron(s)


def move_to_device(batch, device, keys):
    return [batch[key].to(device) for key in keys]


def train_bow(model, loss, optimizer, dataloader, epochs, device="cpu", epoch_evaluation=None):
    dataset_size = len(dataloader.dataset)
    for e in range(epochs):
        epoch_loss_accu = 0
        n_instances = 0
        n_batches = 0
        for batch in dataloader:
            x, y, masks = move_to_device(batch, device, ['x', 'y', 'mask'])

            n_instances += len(batch['x'])
            n_batches += 1

            logits = model.forward(x, masks)
            batch_loss = loss(logits, y)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            epoch_loss_accu += batch_loss.item()

            if (n_batches % 50) == 0:
                print(f"\tloss for the current batch: {batch_loss:>7f}\n"
                      f"\t{n_instances:5d}/{dataset_size:5d} instances processed")
        print(f"epoch{e:3d} done.")
        print(f"average loss per batch: {epoch_loss_accu/n_batches:7f}:")
        if epoch_evaluation:
            print("evaluating on dev")
            dev_score = epoch_evaluation()
            print(f"Epoch evaluation: {dev_score:>7f}")


class LinearModelAccuracy:

    def __init__(self):
        self._accu = 0
        self.n = 0

    def collect_batch(self, logits, y):
        self._accu += torch.sum((logits > 0) == y.bool()).item()
        self.n += len(logits)

    def value(self):
        return self._accu/self.n

    def reset(self):
        self._accu = 0
        self.n = 0


def test_bow(model, measure, dataloader, device):
    measure.reset()
    for batch in dataloader:
        x, y, masks = move_to_device(batch, device, ['x', 'y', 'mask'])
        logits = model.forward(x, masks)
        measure.collect_batch(logits, y)


if __name__ == "__main__":
    # For reproducibility
    torch.manual_seed(SEED)

    # get train and dev
    imdb_full = load_dataset("imdb")
    imdb_full = imdb_full.shuffle(seed=SEED)

    corpus = imdb_full['train']
    if truncate_corpus:
        corpus = imdb_full['train'].select(range(corpus_size))

    # preprocess corpus
    spc_tok = spacy_tokenizer()

    # create the vocabulary and mapping to indices
    voc_ds = Book()
    # add special tokens
    voc_ds.add(UNK)
    voc_ds.add(PAD)

    preprocess = compose(lambda instance: add_words(instance, voc_ds), lambda instance: tokenize(instance, spc_tok),
                         lower, clean_html, clean_extra_spaces)
    corpus = corpus.map(preprocess)

    # split into train-dev (25% dev)
    splits = corpus.train_test_split()
    train = splits['train']
    dev = splits['test']

    # make a pytorch dataloader to feed batches to the train loop
    # wrap the dataset in a pytorch dataset first (and turn inputs to tensors)
    prepare = compose(as_tensor, TextIndexer(voc_ds))
    t_train = DecisionDataset(train, prepare)
    t_dev = DecisionDataset(dev, prepare)

    # make the dataloader
    train_loader = DataLoader(t_train, batch_size=b_size, shuffle=True,
                              collate_fn=lambda instances: pad_batch(instances, voc_ds),
                              pin_memory=True)

    dev_loader = DataLoader(t_dev, batch_size=b_size, shuffle=True,
                            collate_fn=lambda instances: pad_batch(instances, voc_ds),
                            pin_memory=True)

    # model
    stupid = PerceptronOverCombinedWordEmbeddings(nn.Embedding(len(voc_ds), CUSTOM_E_DIM),
                                                  StupidSentenceEmbedding(), Perceptron(CUSTOM_E_DIM))
    # move to gpu
    stupid.to("cuda")
    # loss function
    # (with this sgd should do almost the same as the perceptron algorithm since the loss derivative is 1 at 0)
    per_loss = nn.BCEWithLogitsLoss()

    accuracy_measure = LinearModelAccuracy()


    def eval_on_dev():
        test_bow(stupid, accuracy_measure, dev_loader, device="cuda")
        return accuracy_measure.value()

    adam_opt = torch.optim.Adam(stupid.parameters(), lr=lr)

    print(f"vocabulary size: {len(voc_ds):d}")
    train_bow(stupid, per_loss, adam_opt, train_loader, 20, device="cuda", epoch_evaluation=eval_on_dev)
    torch.save(stupid.state_dict(), os.path.join(SAVE_PATH, f"{MODEL_NAME}.wgt"))
    write_book(voc_ds, SAVE_PATH, MODEL_NAME)
