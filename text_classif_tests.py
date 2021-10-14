import argparse
import os
from datasets import load_dataset
from transformers import BertTokenizerFast, BertModel
import torch
from torch.utils.data import Dataset, DataLoader
from utils import Book, compose, write_book, OccurrencesCount
import torch.nn as nn
from spacy.lang.en import English
from torch.nn.functional import pad
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from multiprocessing import Pool

UNK = "<UNK>"
PAD = "<#>"
SAVE_PATH = "saved_models"
MODEL_NAME = "stupid_one_hot"
SEED = 42
CUSTOM_E_DIM = 64
ID = nn.Identity()


torch_device = "cuda"
truncate_corpus = False
corpus_size = 100
lr = 1e-3
b_size = 64
n_epochs = 7
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


# returns a closure accumulating term occurrences in a corpus
def freq_fn(oc):
    def _cls(words):
        for w in words:
            oc.incr(w)
        return words

    return _cls


# returns a closure accumulating document-occurrences in a corpus
def df_fn(oc):
    def _cls(words):
        for w in set(words):
            oc.incr(w)
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


def as_bow_fn(voc):
    def _cls(words):
        types = set(words)
        indices = sorted([voc[w] for w in types])
        bow_repr = np.array(indices, dtype='i')
        return bow_repr

    return _cls


def tf_fn():
    def _cls(ttype_t):
        oc = OccurrencesCount()
        for idx in ttype_t:
            oc.incr(idx.item())
        return torch.tensor([oc[idx.item()] for idx in ttype_t], dtype=torch.int)

    return _cls


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


def pad_batch(instances, pad_indices, tensor_keys=('x',), y_key="y", mask_pfx="mask_"):
    batch = {}
    for key, pad_idx in zip(tensor_keys, pad_indices):
        max_size = max(instance[key].size()[0] for instance in instances)
        paddings = [(0, max_size - instance[key].size()[0]) for instance in instances]
        padded_instances = [pad(instance[key], padding, mode="constant", value=pad_idx)
                            for (instance, padding) in zip(instances, paddings)]
        instances_batch = torch.stack(padded_instances)
        masks_batch = instances_batch != pad_idx
        batch[key] = instances_batch
        batch[mask_pfx + key] = masks_batch

    labels_batch = torch.stack(gather_by_key(y_key, instances))
    batch[y_key] = labels_batch

    t_keys_set = set(tensor_keys)
    for key in instances[0]:
        if key not in t_keys_set and key != y_key:
            batched_value = gather_by_key(key, instances)
            batch[key] = batched_value

    return batch


def mask_embeddings(w_es, masks):
    return w_es * masks.unsqueeze(2)


class BowEmbedding(nn.Module):

    def __init__(self, voc_size, e_size):
        super(BowEmbedding, self).__init__()
        self.weight = nn.Parameter(torch.zeros(voc_size, e_size, dtype=torch.float))
        nn.init.kaiming_normal_(self.weight)

    def forward(self, one_hot):
        return one_hot.unsqueeze(2) * self.weight


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


def extract_tf(batch_oc):
    return batch_oc / batch_oc.sum(dim=1, keepdim=True)


def extract_idf(instance, voc, df_oc, n_docs):
    batch_ttype_seq = instance['x']
    batch_size, m_len = batch_ttype_seq.shape
    batch_idf = np.array(
        [[np.log(n_docs + 1) - np.log(df_oc[voc.values[idx.item()]] + 1) for idx in batch_ttype_seq[b]] for b in range(batch_size)],
        dtype='f'
    )
    return batch_idf


class TfWeights(nn.Module):
    def __init__(self, voc, to_device):
        super(TfWeights, self).__init__()
        self.voc = voc
        self.to_device = to_device

    def forward(self, w_es, instance):
        tf_w = extract_tf(self.to_device(instance['tf']))
        return tf_w


class TfidfWeights(nn.Module):

    def __init__(self, tf_weights, voc,  df_oc, n_docs, to_device):
        super(TfidfWeights, self).__init__()
        self.df_oc = df_oc
        self.n_docs = n_docs
        self.tf_weights = tf_weights
        self.to_device = to_device
        self.voc = voc

    def forward(self, w_es, instance):
        weights = self.tf_weights(w_es, instance)
        idf_batch = extract_idf(instance, self.voc, self.df_oc, self.n_docs)
        idf_batch = self.to_device(torch.from_numpy(idf_batch))
        tfidf_weights = weights * idf_batch
        return tfidf_weights


class DotProductAttentionWeights(nn.Module):

    def __init__(self, key_s, in_s, to_device, scale=None, nl=ID, one_hot_input=False, masked_item_score=10e-20):
        super(DotProductAttentionWeights, self).__init__()
        self.in_s = in_s
        self.key_p = torch.nn.Parameter(torch.zeros(key_s, dtype=torch.float))
        self.query_tfm = nn.Linear(in_s, key_s)
        if one_hot_input:
            self.query_tfm = nn.Embedding.from_pretrained(self.query_tfm.weight, freeze=False)
        self.nl = nl
        self.norm = nn.Softmax(dim=1)
        self.scale = scale
        if not self.scale:
            self.scale = np.sqrt(in_s)
        self.to_device = to_device
        self.m_score = masked_item_score

    def forward(self, w_es, instance):
        # has shape batch_size, len_ttype_seq, key_s, 1
        query = self.query_tfm(w_es).unsqueeze(3)
        # has shape 1, key_s
        key = self.key_p.unsqueeze(0)
        # has shape batch_size, len_ttype_seq
        scores = (self.nl(torch.matmul(key, query))/self.scale).squeeze(2, 3)
        mask = self.to_device(instance['mask_x'])*self.m_score
        masked_scores = scores * mask
        return self.norm(masked_scores)


class WeightedAverage(nn.Module):

    def __init__(self, weighting_scheme):
        super(WeightedAverage, self).__init__()
        self.weighting_scheme = weighting_scheme

    def forward(self, w_es, instance):
        weights = self.weighting_scheme(w_es, instance).unsqueeze(2)
        average = torch.sum(weights * w_es, dim=1)
        return average, weights


class OneHotWeightedAverage(nn.Module):
    def __init__(self, v_size, to_device, weighting_scheme):
        super(OneHotWeightedAverage, self).__init__()
        self.v_size = v_size
        self.weighting_scheme = weighting_scheme
        self.to_device = to_device

    def __call__(self, w_es, instance):
        weights = self.weighting_scheme(w_es, instance)
        w_a = self.to_device(torch.zeros(w_es.shape[0], self.v_size))
        for b in range(w_a.shape[0]):
            for i, idx in enumerate(instance['x'][b]):
                w_a[b][idx.item()] = weights[b][i]

        return w_a, weights


class BowModel(nn.Module):

    def __init__(self, embedding, composition, perceptron):
        super(BowModel, self).__init__()
        self.perceptron = perceptron
        self.embedding = embedding
        self.composition = composition

    def forward(self, bow, instance):
        w_es = self.embedding(bow)
        # masked_w_es = mask_embeddings(w_es, masks)
        s_e, composition_weights = self.composition(w_es, instance)
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
            x, y = move_to_device(batch, device, ['x', 'y'])

            n_instances += len(batch['x'])
            n_batches += 1

            logits, composition_weights = model.forward(x, batch)
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
        x, y = move_to_device(batch, device, ['x', 'y'])
        logits, composition_weights = model.forward(x, batch)
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
                        help="composition type (tf | attention | tf-idf)",
                        default="tf")

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

    # occurrence counters for df and global frequency
    d_oc = OccurrencesCount()
    g_oc = OccurrencesCount()

    # preprocessing functions to clean, tokenize, build the vocabulary and compute tf, idf and global frequency info
    clean_text = InstanceUpdate("text", "cleaned_text", simple_cleanup)
    add_tokens = InstanceUpdate("cleaned_text", "words", spc_tok)
    build_vocab = InstanceUpdate("words", "words", add_to_voc_fn(voc_train))
    sub_unknown = InstanceUpdate("words", "words", sub_unknown_fn(voc_train))
    add_df = InstanceUpdate("words", "words", df_fn(d_oc))
    add_gf = InstanceUpdate("words", "words", freq_fn(g_oc))

    preprocess_train = compose(add_df, add_gf, build_vocab, add_tokens, clean_text)
    preprocess_test = compose(sub_unknown, add_tokens, clean_text)

    train = train.map(preprocess_train)
    test = test.map(preprocess_test)
    n_docs_train = len(train)

    dev = None
    if dev_split_proportion > 0:
        # split train into train-dev
        splits = train.train_test_split(dev_split_proportion)
        train = splits['train']
        dev = splits['test']

    # make a pytorch dataloader to feed batches to the train loop
    # wrap the dataset in a pytorch dataset first (and turn inputs to tensors)
    as_indices = as_indices_fn(voc_train)
    add_z = InstanceUpdate("words", "z", compose(torch.as_tensor, as_indices))
    add_x = InstanceUpdate("words", "x", compose(torch.as_tensor, as_bow_fn(voc_train)))
    add_y = InstanceUpdate("label", "y", compose(torch.as_tensor, as_float_array))
    add_tf = InstanceUpdate("x", "tf", tf_fn())

    add_signal = compose(add_tf, add_x, add_y, add_z)
    t_train = DecisionDataset(train, add_signal)
    t_dev = DecisionDataset(dev, add_signal)
    t_test = DecisionDataset(test, add_signal)

    # make the dataloader
    def collate_fn(instances):
        return pad_batch(instances, [voc_train[PAD]]*2 + [0], tensor_keys=['x', 'z', 'tf'],
                         y_key='y')

    train_loader = DataLoader(t_train, batch_size=b_size, shuffle=True,
                              collate_fn=collate_fn,
                              pin_memory=True)

    dev_loader = None
    if dev:
        dev_loader = DataLoader(t_dev, batch_size=b_size, shuffle=True,
                                collate_fn=collate_fn,
                                pin_memory=True)

    test_loader = DataLoader(t_test, batch_size=b_size, shuffle=True,
                             collate_fn=collate_fn,
                             pin_memory=True)

    # model
    h_size = CUSTOM_E_DIM
    composition_layer_factory = WeightedAverage
    use_one_hot = args.input == "one-hot"
    # word embedding type
    if use_one_hot:
        h_size = len(voc_train)

        def composition_layer_factory(w_scheme):
            return OneHotWeightedAverage(len(voc_train), weighting_scheme=w_scheme, to_device=lambda x: x.to(torch_device))

        e_layer = ID

    elif args.input == "learned":
        e_layer = nn.Embedding(len(voc_train), CUSTOM_E_DIM)

    else:
        raise NotImplementedError

    # words weighting scheme
    if args.composition == "tf":
        weights_scheme = TfWeights(voc_train, lambda x: x.to(torch_device))

    elif args.composition == "attention":
        weights_scheme = DotProductAttentionWeights(in_s=h_size, key_s=CUSTOM_E_DIM,
                                                    one_hot_input=use_one_hot, to_device=lambda x: x.to(torch_device))

    elif args.composition == "tf-idf":
        weights_scheme = TfidfWeights(TfWeights(voc_train, lambda x: x.to(torch_device)),
                                      voc_train, d_oc, n_docs=n_docs_train,
                                      to_device=lambda x: x.to(torch_device))

    else:
        raise ValueError("{:s} is not a known composition method ".format(args.composition))

    composition_fn = composition_layer_factory(weights_scheme)

    # output layer
    if args.model == "perceptron":
        m_type = Perceptron

    else:
        raise NotImplementedError

    bow_model = BowModel(e_layer, composition_fn, m_type(h_size))
    # move to device
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
