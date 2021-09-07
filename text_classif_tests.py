import argparse
import os
from datasets import load_dataset
from transformers import BertTokenizerFast, BertModel
import torch
from torch.utils.data import Dataset, DataLoader
from utils import Book, compose, write_book
import torch.nn as nn
from spacy.lang.en import English
from torch.nn.functional import pad
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

UNK = "<UNK>"
PAD = "<#>"
SAVE_PATH = "saved_models"
MODEL_NAME = "stupid"
SEED = 42
CUSTOM_E_DIM = 1024
ID = nn.Identity()


torch_device = "cuda"
truncate_corpus = False
corpus_size = 1000
lr = 1e-3
b_size = 64
dev_split_proportion = 0.25

tag_re = re.compile("</?\\w+(?:\\s+\\w+=\"[^\\s\"]+\")*\\s*/?>")
extra_spaces = re.compile("\\s+")


# tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
# model = BertModel.from_pretrained("bert-base-cased")

def clean_extra_spaces(text):
    return extra_spaces.sub(" ", text)


def clean_html(text):
    return tag_re.sub(" ", text)


def lower(text):
    return text.lower()


def spacy_tokenizer():
    spacy_proc = English()
    tok = spacy_proc.tokenizer
    # Need to convert to list to avoid weird bug with pyarrow
    # when directly storing the spacy iterable inside a dictionary
    return lambda t: [str(w) for w in tok(t)]


# returns a closure adding words to a vocabulary (and returning the words unchanged)
def add_to_voc_fn(voc):
    def _cls(words):
        for w in words:
            voc.add(w)
        return words

    return _cls


# returns a closure replacing unknown words in the input
def sub_unknown_fn(voc, unk=UNK):
    def _sub(w):
        if w in voc:
            return w
        else:
            return unk

    def _cls(words):
        return [_sub(w) for w in words]

    return _cls


# returns a closure turning words into a list of indices (relative to some indexing voc)
def as_indices_fn(voc):
    def _cls(words):
        return np.array([voc[w] for w in words], dtype=int)

    return _cls


def as_float_array(label):
    return np.array([label], dtype=np.float32)


simple_cleanup = compose(clean_extra_spaces, clean_html, lower)


# Factory for a closure applying a treatment to some field of an instance and storing the result into a specified field
class InstanceUpdate:
    def __init__(self, src_key, tgt_key, action):
        self._src_key = src_key
        self._tgt_key = tgt_key
        self._action = action

    def __call__(self, instance):
        instance.update({self._tgt_key: self._action(instance[self._src_key])})
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


def pad_batch(instances, voc, x_key="x", y_key="y", m_key="mask"):
    max_size = max(instance[x_key].size()[0] for instance in instances)
    paddings = [(0, max_size - instance[x_key].size()[0]) for instance in instances]
    padded_instances = [pad(instance[x_key], padding, mode="constant", value=voc[PAD])
                        for (instance, padding) in zip(instances, paddings)]
    instances_batch = torch.stack(padded_instances)
    labels_batch = torch.stack(gather_by_key(y_key, instances))
    masks_batch = instances_batch != voc[PAD]
    batch = {x_key: instances_batch, y_key: labels_batch, m_key: masks_batch}
    for key in instances[0]:
        if key != x_key and key != y_key:
            batched_value = gather_by_key(key, instances)
            batch[key] = batched_value

    return batch


class Perceptron(nn.Module):

    def __init__(self, in_s, nl=ID):
        super(Perceptron, self).__init__()
        self.in_s = in_s
        self.n_lin = nl
        self.lin = nn.Linear(in_s, 1)

    def forward(self, x, *args, **kargs):
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


class LinearModelF1:

    def __init__(self, device):
        self._tp = 0
        self._tn = 0
        self._fp = 0
        self._fn = 0
        self.n = 0
        self.device = device

    def collect_batch(self, logits, y):
        size = len(logits)
        top = torch.ones(size).unsqueeze(1).int().to(self.device)
        preds = (logits > 0).int()
        tgt = y.int()
        neg_preds = top - preds
        neg_tgt = top - tgt
        tp = torch.sum(preds * tgt).item()
        tn = torch.sum(neg_preds * neg_tgt).item()
        fp = torch.sum(preds * neg_tgt).item()
        fn = torch.sum(neg_preds * tgt).item()
        self._tp += tp
        self._tn += tn
        self._fp += fp
        self._fn += fn
        self.n += size

    def value(self):
        if self.n != (self._tp + self._tn + self._fp + self._fn):
            raise ValueError
        prec_deno = (self._tp + self._fp)
        rec_deno = (self._tp + self._fn)
        if prec_deno == 0 or rec_deno == 0 or self._tp == 0:
            return 0
        else:
            prec = self._tp / prec_deno
            rec = self._tp / rec_deno
            return 2 * prec * rec / (prec + rec)

    def reset(self):
        self._tp = 0
        self._fp = 0
        self._tn = 0
        self._fn = 0
        self.n = 0


def test_bow(model, measure, dataloader, device):
    measure.reset()
    for batch in dataloader:
        x, y, masks = move_to_device(batch, device, ['x', 'y', 'mask'])
        logits = model.forward(x, masks)
        measure.collect_batch(logits, y)


def iter_by_key(dataset, key):
    for instance in dataset:
        yield instance[key]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='train a simple binary classifier on the IMDB dataset')
    parser.add_argument("--input-type", dest="input", help="type of input (tf-idf | glove | learned)",
                        default="learned")
    parser.add_argument("--model-type", dest="model", help="model type (perceptron | linear svm | svm)",
                        default="perceptron")

    args = parser.parse_args()

    # For reproducibility
    torch.manual_seed(SEED)

    # get train and dev
    imdb = load_dataset("imdb")
    imdb = imdb.shuffle(seed=SEED)

    train = imdb['train']
    test = imdb['test']

    # to debug faster
    if truncate_corpus:
        train = train.select(range(corpus_size))
        test = test.select(range(corpus_size))

    # preprocess corpus
    spc_tok = spacy_tokenizer()

    # create the vocabulary and mapping to indices
    voc_train = Book()
    # add special tokens
    voc_train.add(UNK)
    voc_train.add(PAD)

    # clean, tokenize and build the vocabulary
    clean_text = InstanceUpdate("text", "cleaned_text", simple_cleanup)
    add_tokens = InstanceUpdate("cleaned_text", "words", spc_tok)
    build_vocab = InstanceUpdate("words", "words", add_to_voc_fn(voc_train))
    sub_unknown = InstanceUpdate("words", "words", sub_unknown_fn(voc_train))

    preprocess_train = compose(build_vocab, add_tokens, clean_text)
    preprocess_test = compose(sub_unknown, add_tokens, clean_text)

    train = train.map(preprocess_train)
    test = test.map(preprocess_test)

    if args.input == "tf-idf":

        vectorizer = TfidfVectorizer(vocabulary=voc_train, tokenizer=spc_tok, token_pattern=None, dtype=np.float32)
        vectorizer.fit(iter_by_key(train, "cleaned_text"))
        # add the document vector to each instance

        def as_tfidf_vector(t):
            x = vectorizer.transform([t])
            return torch.as_tensor(x.toarray()[0])

        add_x = InstanceUpdate("cleaned_text", "x", as_tfidf_vector)
    else:
        add_x = InstanceUpdate("words", "x", compose(torch.as_tensor, as_indices_fn(voc_train)))

    dev = None
    if dev_split_proportion > 0:
        # split train into train-dev
        splits = train.train_test_split(dev_split_proportion)
        train = splits['train']
        dev = splits['test']

    # make a pytorch dataloader to feed batches to the train loop
    # wrap the dataset in a pytorch dataset first (and turn inputs to tensors)
    add_y = InstanceUpdate("label", "y", compose(torch.as_tensor, as_float_array))

    add_signal = compose(add_y, add_x)
    t_train = DecisionDataset(train, add_signal)
    t_dev = DecisionDataset(dev, add_signal)
    t_test = DecisionDataset(test, add_signal)

    # make the dataloader
    train_loader = DataLoader(t_train, batch_size=b_size, shuffle=True,
                              collate_fn=lambda instances: pad_batch(instances, voc_train),
                              pin_memory=True)

    dev_loader = None
    if dev:
        dev_loader = DataLoader(t_dev, batch_size=b_size, shuffle=True,
                                collate_fn=lambda instances: pad_batch(instances, voc_train),
                                pin_memory=True)

    test_loader = DataLoader(t_test, batch_size=b_size, shuffle=True,
                             collate_fn=lambda instances: pad_batch(instances, voc_train),
                             pin_memory=True)

    # model
    if args.model == "perceptron" and args.input == "learned":
        bow_model = PerceptronOverCombinedWordEmbeddings(nn.Embedding(len(voc_train), CUSTOM_E_DIM),
                                                         StupidSentenceEmbedding(), Perceptron(CUSTOM_E_DIM))
    elif args.model == "perceptron" and args.input == "tf-idf":
        bow_model = Perceptron(len(voc_train))

    else:
        raise NotImplementedError
    # move to gpu
    bow_model.to(torch_device)
    # loss function
    # (with this sgd should do almost the same as the perceptron algorithm since the loss derivative is 1 at 0)
    per_loss = nn.BCEWithLogitsLoss()

    accuracy_measure = LinearModelF1(torch_device)
    # accuracy_measure = LinearModelAccuracy()

    if dev:
        def eval_on_dev():
            test_bow(bow_model, accuracy_measure, dev_loader, device=torch_device)
            return accuracy_measure.value()
    else:
        eval_on_dev = None

    adam_opt = torch.optim.Adam(bow_model.parameters(), lr=lr)

    print(f"vocabulary size: {len(voc_train):d}")
    train_bow(bow_model, per_loss, adam_opt, train_loader, 10, device=torch_device, epoch_evaluation=eval_on_dev)
    torch.save(bow_model.state_dict(), os.path.join(SAVE_PATH, f"{MODEL_NAME}.wgt"))
    write_book(voc_train, SAVE_PATH, MODEL_NAME)

    print("evaluating model: ")
    test_bow(bow_model, accuracy_measure, test_loader, device=torch_device)
    print(accuracy_measure.value())
