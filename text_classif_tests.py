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
CUSTOM_E_DIM = 128
ID = nn.Identity()


torch_device = "cuda"
truncate_corpus = False
corpus_size = 1000
lr = 1e-3
b_size = 2
n_epochs = 6
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


def mask_embeddings(w_es, masks):
    return w_es * masks.unsqueeze(2)


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


class StupidWeights(nn.Module):

    def __init__(self, pad_index, to_device):
        super(StupidWeights, self).__init__()
        self.to_device = to_device
        self.pad_index = torch.tensor(pad_index)

    def forward(self, indices, w_es):
        weights = torch.ne(indices, self.to_device(self.pad_index))
        s_lengths = torch.sum(weights, dim=1).unsqueeze(1)
        weights = weights/s_lengths
        return weights


class FFAttentionWeights(nn.Module):

    def __init__(self, in_s, nl_factory=nn.Tanh):
        super(FFAttentionWeights, self).__init__()
        self.in_s = in_s
        self.score_lin = nn.Linear(self.in_s, 1)
        self.score_nl = nl_factory()
        self.score_tfm = nn.Sequential(self.score_lin, self.score_nl)
        self.norm = nn.Softmax(dim=1)

    def forward(self, indices, w_es):
        # has shape batch_size, seq_length
        scores = self.score_tfm(w_es).squeeze(2)
        return self.norm(scores)


class DotProductAttentionWeights(nn.Module):

    def __init__(self, in_s):
        super(DotProductAttentionWeights, self).__init__()
        self.in_s = in_s
        self.key_p = torch.nn.Parameter(torch.zeros(in_s, dtype=torch.float))
        self.query_tfm = torch.nn.Linear(in_s, in_s)
        self.norm = nn.Softmax(dim=1)

    def forward(self, indices, w_es):
        # has shape batch_size, seq_length, in_s, 1
        query = self.query_tfm(w_es).unsqueeze(3)
        # has shape 1, in_s
        key = self.key_p.unsqueeze(0)
        # has shape batch_size, seq_length
        scores = (torch.matmul(key, query)/np.sqrt(self.in_s)).squeeze(2, 3)
        return self.norm(scores)


class TfidfWeights:

    def __init__(self, tfidf_vectorizer, to_device):
        self.v = tfidf_vectorizer
        self.to_device = to_device

    def __call__(self, indices, w_es):
        tfidf_m = self.v.transform(indices).toarray()
        s = indices.shape
        weights = torch.zeros(*s, dtype=torch.float, requires_grad=False)
        for i in range(s[0]):
            for j in range(s[1]):
                weights = tfidf_m[i][indices[i][j]]

        return self.to_device(weights)


class WeightedAverage(nn.Module):

    def __init__(self, weighting_scheme):
        super(WeightedAverage, self).__init__()
        self.weighting_scheme = weighting_scheme

    def forward(self, indices, w_es):
        weights = self.weighting_scheme(indices, w_es).unsqueeze(2)
        average = torch.sum(weights * w_es, dim=1)
        return average, weights


def one_hot(index, voc_size):
    oh = torch.zeros(voc_size)
    oh[index] = 1
    return oh


class OneHotWeightedAverage(nn.Module):

    def __init__(self, weighting_scheme, to_device):
        super(OneHotWeightedAverage, self).__init__()
        self.weighting_scheme = weighting_scheme
        self.to_device = to_device

    def forward(self, indices, w_es):
        (batch_size, voc_size) = indices.shape
        # shape batch_size * seq_length
        weights = self.weighting_scheme(indices, w_es)
        # shape seq_length * batch_size
        weights = torch.transpose(weights, 0, 1)
        average = self.to_device(torch.zeros(batch_size, voc_size), dtype=torch.float)
        for i in range(voc_size):
            # tmp is batch_size * voc_size
            tmp = self.to_device(torch.stack(
                [one_hot(indices[b][i].item(), voc_size) for b in range(batch_size)],
                dim=0
            ))
            # append a batch_size * voc_size tensor to average
            average += weights[i].unsqueeze(1) * tmp
        return average, weights


class BOWModel(nn.Module):

    def __init__(self, embedding, composition, perceptron):
        super(BOWModel, self).__init__()
        self.perceptron = perceptron
        self.embedding = embedding
        self.composition = composition

    def forward(self, indices, masks):
        w_es = self.embedding(indices)
        masked_w_es = mask_embeddings(w_es, masks)
        s_e, composition_weights = self.composition(indices, masked_w_es)
        return self.perceptron(s_e), composition_weights


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

            logits, composition_weights = model.forward(x, masks)
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
        logits, composition_weights = model.forward(x, masks)
        measure.collect_batch(logits, y)


def iter_by_key(dataset, key):
    for instance in dataset:
        yield instance[key]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='train a simple binary classifier on the IMDB dataset')
    parser.add_argument("--embedding-type", dest="input", help="type of input (one-hot | glove | learned)",
                        default="learned")
    parser.add_argument("--model-type", dest="model", help="model type (perceptron | linear svm | svm)",
                        default="perceptron")
    parser.add_argument("--sentence-composition", dest="composition",
                        help="composition type (naive | ff_attention |  sdp_attention | tf-idf)",
                        default="naive")

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
    composition_layer_factory = WeightedAverage
    # word embedding type
    if args.input == "one-hot":
        h_size = len(voc_train)

        # leave embeddings as is (that is indices)
        # because one-hot stuff takes too much space
        e_layer = ID

        # so, need to use a special averaging class to handle the 'implicit' one-hot repr
        def composition_layer_factory(ws):
            return OneHotWeightedAverage(ws, lambda x: x.to(torch_device))

    elif args.input == "learned":
        h_size = CUSTOM_E_DIM
        e_layer = nn.Embedding(len(voc_train), CUSTOM_E_DIM)

    else:
        raise NotImplementedError

    # words weighting scheme
    if args.composition == "naive":
        weights_scheme = StupidWeights(voc_train[PAD], lambda x: x.to(torch_device))

    elif args.composition == "ff_attention":
        weights_scheme = FFAttentionWeights(CUSTOM_E_DIM)

    elif args.composition == "sdp_attention":
        weights_scheme = DotProductAttentionWeights(CUSTOM_E_DIM)

    elif args.composition == "tf-idf":
        vectorizer = TfidfVectorizer(vocabulary=voc_train, tokenizer=spc_tok, token_pattern=None, dtype=np.float32)
        vectorizer.fit(iter_by_key(train, "cleaned_text"))
        weights_scheme = TfidfWeights(vectorizer, lambda x: x.to(torch_device))

    else:
        raise ValueError("{:s} is not a known composition method ".format(args.composition))

    composition_fn = composition_layer_factory(weights_scheme)

    # output layer
    if args.model == "perceptron":
        m_type = Perceptron

    else:
        raise NotImplementedError

    bow_model = BOWModel(e_layer, composition_fn, m_type(h_size))
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
    train_bow(bow_model, per_loss, adam_opt, train_loader, n_epochs, device=torch_device, epoch_evaluation=eval_on_dev)
    torch.save(bow_model.state_dict(), os.path.join(SAVE_PATH, f"{MODEL_NAME}.wgt"))
    write_book(voc_train, SAVE_PATH, MODEL_NAME)

    print("evaluating model: ")
    test_bow(bow_model, accuracy_measure, test_loader, device=torch_device)
    print(accuracy_measure.value())
