import argparse
import os

from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from utils import Book, compose, write_book, OccurrencesCount
import torch.nn as nn
from spacy.lang.en import English
from torch.nn.functional import pad
import re
import numpy as np
import copy


UNK = "<UNK>"
PAD = "<#>"
SAVE_PATH = "saved_models"
MODEL_NAME = "unknown_model"
SEED = 42
CUSTOM_E_DIM = 200
ID = nn.Identity()
DEVICE = "cuda"
TRUNCATE = False
TRUNC_SIZE = 100
LR = 0.001
B_SIZE = 512
N_EPOCHS = 10
DEV_PROP = 0.25
OUTPUT_DEV = True

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


# returns a closure replacing words with UNK if they occur less than threshold times
def make_low_frequency_unknown_fn(g_oc, threshold=2,  unk=UNK):
    def _sub(w):
        if g_oc[w] < threshold:
            return unk
        else:
            return w

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


def load_embeddings(emb_dict, path_to_emb_file, voc):

    def read_embs(fdesc):
        while True:
            line = fdesc.readline()
            if line:
                entry = line.split(" ")
                w = entry[0]
                try:
                    embedding = np.array([float(component) for component in entry[1:]], dtype=float)
                    yield w, embedding
                except Exception:
                    print(line)
            else:
                break

    with open(path_to_emb_file) as emb_desc:
        for word, word_embedding in read_embs(emb_desc):
            voc.add(word)
            emb_dict[word] = word_embedding


def compute_mean_embedding(emb_dict):
    return np.mean(list(emb_dict.values()), axis=0)


def add_special_embs(emb_dict, spc_toks, spc_embs):
    for t, e in zip(spc_toks, spc_embs):
        emb_dict[t] = e


def query_embedding(emb_dict, word, factory):
    if word not in emb_dict:
        e = factory()
        emb_dict[word] = e
    else:
        e = emb_dict[word]
    return e


def make_embedding_tensor(emb_dict, voc, factory):
    if emb_dict:
        weights = np.stack([query_embedding(emb_dict, w, factory) for w in voc.values])
        return weights
    else:
        return None


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

    def __init__(self, in_s, h_s, nl=torch.nn.ReLU, n_layers=3):
        super(Perceptron, self).__init__()
        self.in_s = in_s
        self.h_s = h_s
        self.n_lin = nl()
        self.n_layers = n_layers
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(self.in_s, self.h_s))
        self.dropout = nn.Dropout(0.5)
        self.layers.extend(
            [torch.nn.Linear(self.h_s, self.h_s) for _ in range(self.n_layers - 1)]
        )
        self.out_layer = nn.Linear(self.h_s, 1)

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = self.dropout(self.n_lin(layer(h)))
        return self.out_layer(h)


class WeightingScheme(nn.Module):

    def forward(self, indices, w_es, instance):
        pass


def extract_tf(batch_oc):
    return batch_oc / batch_oc.sum(dim=1, keepdim=True)


class OnesW(WeightingScheme):
    def __init__(self, to_device):
        super(OnesW, self).__init__()
        self.to_device = to_device

    def forward(self, indices, w_es, instance):
        return self.to_device(torch.ones(*indices.shape, dtype=torch.int))


class TfWeights(WeightingScheme):
    def __init__(self, voc, to_device):
        super(TfWeights, self).__init__()
        self.voc = voc
        self.to_device = to_device

    def forward(self, indices, w_es, instance):
        tf_w = extract_tf(self.to_device(instance['tf']))
        return tf_w


class TfidfWeights(WeightingScheme):

    def __init__(self, tf_weights, idf_emb):
        super(TfidfWeights, self).__init__()
        self.idf_emb = idf_emb
        self.tf_weights = tf_weights

    def forward(self, indices, w_es, instance):
        weights = self.tf_weights(indices, w_es, instance)
        idf_batch = self.idf_emb(indices).squeeze(2)
        tfidf_weights = weights * idf_batch
        return tfidf_weights


class DotProductAttentionWeights(WeightingScheme):

    def __init__(self, key_s, in_s, to_device, scale=None, nl=ID, one_hot_input=False, masked_item_score=10e-20):
        super(DotProductAttentionWeights, self).__init__()
        self.in_s = in_s
        self.key_s = key_s
        self.to_device = to_device
        self.key_p = torch.nn.Parameter(torch.zeros(self.key_s, dtype=torch.float))
        if one_hot_input:
            self.query_tfm = self.to_device(nn.Embedding(self.in_s, self.key_s))
        else:
            self.query_tfm = self.to_device(nn.Linear(self.in_s, self.key_s))
        self.nl = nl
        self.norm = nn.Softmax(dim=1)
        self.scale = scale
        if not self.scale:
            self.scale = np.sqrt(key_s)
        self.m_score = masked_item_score

    def forward(self, indices, w_es, instance):

        # has shape batch_size, len_ttype_seq, key_s, 1
        query = self.query_tfm(w_es).unsqueeze(3)
        # has shape 1, key_s
        key = self.key_p.unsqueeze(0)
        # has shape batch_size, len_ttype_seq
        scores = (self.nl(torch.matmul(key, query))/self.scale).squeeze(2).squeeze(2)
        mask = self.to_device(instance['mask_x'])*self.m_score
        masked_scores = scores * mask
        return self.norm(masked_scores)


class WeightedAverage(nn.Module):

    def __init__(self, weighting_scheme, values_tfm=None):
        super(WeightedAverage, self).__init__()
        self.weighting_scheme = weighting_scheme
        self.values_tfm = values_tfm
        if not self.values_tfm:
            self.values_tfm = ID

    def forward(self, indices, w_es, instance):
        weights = self.weighting_scheme(indices, w_es, instance).unsqueeze(2)
        values = self.values_tfm(w_es)
        average = torch.sum(weights * values, dim=1)
        return average, weights


class OneHotWeightedAverage(nn.Module):
    def __init__(self, v_size, to_device, weighting_scheme):
        super(OneHotWeightedAverage, self).__init__()
        self.v_size = v_size
        self.weighting_scheme = weighting_scheme
        self.to_device = to_device

    def __call__(self, indices, w_es, instance):
        weights = self.weighting_scheme(indices, w_es, instance)
        w_a = self.to_device(torch.zeros(w_es.shape[0], self.v_size))
        for b in range(w_a.shape[0]):
            for i, idx in enumerate(instance['x'][b]):
                w_a[b][idx.item()] = weights[b][i]

        return w_a, weights


class BiLSTMEncoding(nn.Module):

    def __init__(self, e_size, h_size, to_device, num_layers=3):
        super(BiLSTMEncoding, self).__init__()
        self.e_size = e_size
        self.h_size = h_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=self.e_size, hidden_size=self.h_size,
                            bidirectional=True, batch_first=False, num_layers=self.num_layers, dropout=0.5)
        self.to_device = to_device

    def forward(self, indices, w_es, instance):
        toks = self.to_device(w_es).transpose(0, 1).contiguous()
        lstm_out, (hn, cn) = self.lstm(toks)
        last_layer_index = 2 * (self.num_layers - 1)
        s_encoding = torch.cat((hn[last_layer_index], hn[last_layer_index + 1]), dim=1)
        s_encoding = s_encoding
        return s_encoding, None


class BowModel(nn.Module):

    def __init__(self, embedding, composition, perceptron):
        super(BowModel, self).__init__()
        self.perceptron = perceptron
        self.embedding = embedding
        self.composition = composition

    def forward(self, bow, instance):
        w_es = self.embedding(bow)
        s_e, composition_weights = self.composition(bow, w_es, instance)
        return self.perceptron(s_e), composition_weights


def move_to_device(batch, device, keys):
    return [batch[key].to(device) for key in keys]


def train_bow(model, loss, optimizer, dataloader, epochs, device="cpu", epoch_evaluation=None, early_stop=True):
    dataset_size = len(dataloader.dataset)
    # can only early stop if there is a dev set and evaluation
    early_stop = early_stop & (epoch_evaluation is not None)
    current_eval = float("-inf")
    current_loss = float("+inf")
    current_state = copy.deepcopy(model.state_dict())
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

            loss_value = batch_loss.item()
            epoch_loss_accu += loss_value

            if (n_batches % 50) == 0:
                print(f"\tloss for the current batch: {batch_loss:>7f}\n"
                      f"\t{n_instances:5d}/{dataset_size:5d} instances processed")
        print(f"epoch{e:3d} done.")
        print(f"average loss per batch: {epoch_loss_accu/n_batches:7f}:")
        if epoch_evaluation:
            model.eval()
            print("evaluating on dev")
            dev_score = epoch_evaluation()
            print(f"Epoch evaluation: {dev_score:>7f}")
            model.train()
            if early_stop & (current_loss > epoch_loss_accu) & (dev_score < current_eval):
                model.load_state_dict(current_state)
                break
            current_eval = dev_score
            current_loss = epoch_loss_accu
            current_state = copy.deepcopy(model.state_dict())


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
    with torch.no_grad():
        for batch in dataloader:
            x, y = move_to_device(batch, device, ['x', 'y'])
            logits, composition_weights = model.forward(x, batch)
            measure.collect_batch(logits, y)


def iter_by_key(dataset, key):
    for instance in dataset:
        yield instance[key]


def prepare_corpora(config, seed, voc):
    # get train and dev
    imdb = load_dataset("imdb")
    imdb = imdb.shuffle(seed=seed)

    train = imdb['train']
    test = imdb['test']

    # to debug faster
    if config.truncate:
        train = train.select(range(config.trunc_size))
        test = test.select(range(config.trunc_size))

    # preprocess corpus
    spc_tok = spacy_tokenizer()

    # occurrence counters for df and global frequency
    df_oc = OccurrencesCount()
    g_oc = OccurrencesCount()

    # preprocessing functions to clean, tokenize, build the vocabulary and compute tf, idf and global frequency info
    clean_text = InstanceUpdate("text", "cleaned_text", simple_cleanup)
    add_tokens = InstanceUpdate("cleaned_text", "words", spc_tok)
    build_vocab = InstanceUpdate("words", "words", add_to_voc_fn(voc))
    sub_unknown = InstanceUpdate("words", "words", sub_unknown_fn(voc))
    add_df = InstanceUpdate("words", "words", df_fn(df_oc))
    add_gf = InstanceUpdate("words", "words", freq_fn(g_oc))
    make_lfu = InstanceUpdate("words", "words", make_low_frequency_unknown_fn(g_oc))

    preprocess_train = compose(add_df, add_gf, add_tokens, clean_text)
    preprocess_test = compose(add_tokens, clean_text)

    train = train.map(preprocess_train)
    test = test.map(preprocess_test)

    # Sub infrequent with UNK
    train = train.map(make_lfu)
    test = test.map(make_lfu)

    # build vocabulary
    train = train.map(build_vocab)

    # replace oov with UNK
    train = train.map(sub_unknown)
    test = test.map(sub_unknown)

    dev = None
    if config.dev_prop > 0:
        # split train into train-dev
        splits = train.train_test_split(config.dev_prop)
        train = splits['train']
        dev = splits['test']
    return train, dev, test, voc, df_oc, g_oc


def preprocess(config, seed):
    voc = Book()
    voc.add(UNK)
    voc.add(PAD)

    emb_dict = None
    factory = None
    if config.we_file:
        emb_dict = {}
        load_embeddings(emb_dict, config.we_file, voc)
        unk_emb = compute_mean_embedding(emb_dict)
        pad_emb = np.zeros(config.h_size, dtype=np.float32)
        add_special_embs(emb_dict, [UNK, PAD], [unk_emb, pad_emb])

        def factory():
            return unk_emb

    train, dev, test, voc, df, gf = prepare_corpora(config, seed, voc)
    embeddings = make_embedding_tensor(emb_dict, voc, factory)

    return train, dev, test, voc, df, gf, embeddings


def make_dataloader(voc, train, dev, test, config):
    # make a pytorch dataloader to feed batches to the train loop
    # wrap the dataset in a pytorch dataset first (and turn inputs to tensors)
    as_indices = as_indices_fn(voc)
    if config.composition == "lstm":
        x_fn = as_indices
    else:
        x_fn = as_bow_fn(voc)
    add_x = InstanceUpdate("words", "x", compose(torch.as_tensor, x_fn))
    add_y = InstanceUpdate("label", "y", compose(torch.as_tensor, as_float_array))
    add_tf = InstanceUpdate("x", "tf", tf_fn())

    add_signal = compose(add_tf, add_x, add_y)
    t_train = DecisionDataset(train, add_signal)
    t_dev = DecisionDataset(dev, add_signal)
    t_test = DecisionDataset(test, add_signal)

    # make the dataloader
    def collate_fn(instances):
        return pad_batch(instances, [voc[PAD], 0], tensor_keys=['x', 'tf'],
                         y_key='y')

    print(config.batch_size)
    train_loader = DataLoader(t_train, batch_size=config.batch_size, shuffle=True,
                              collate_fn=collate_fn,
                              pin_memory=True)

    dev_loader = None
    if dev:
        dev_loader = DataLoader(t_dev, batch_size=config.batch_size, shuffle=True,
                                collate_fn=collate_fn,
                                pin_memory=True)

    test_loader = DataLoader(t_test, batch_size=config.batch_size, shuffle=True,
                             collate_fn=collate_fn,
                             pin_memory=True)
    return train_loader, dev_loader, test_loader


def build_model(config, voc, df_oc, device, n_docs_train, embeddings=None):
    # model
    def composition_layer_factory(w_scheme):
        values_tfm = None
        # values_tfm = torch.nn.Linear(config.h_size, config.h_size)
        # values_tfm.weight = torch.nn.Parameter(torch.diag(torch.ones(config.h_size)))
        return WeightedAverage(w_scheme, values_tfm=values_tfm)
    use_one_hot = config.input == "one-hot"
    h_size = config.h_size
    # default embedding size to the size of other hidden layers
    e_size = h_size
    # word embedding type
    if use_one_hot:
        # with one_hot embeddings, embedding size is the vocabulary size
        e_size = len(voc)

        def composition_layer_factory(w_scheme):
            return OneHotWeightedAverage(e_size, weighting_scheme=w_scheme,
                                         to_device=lambda x: x.to(device))

        e_layer = ID

    elif config.input == "learned":
        if embeddings is None:
            e_layer = nn.Embedding(len(voc), e_size)
        else:
            e_layer = nn.Embedding.from_pretrained(torch.from_numpy(embeddings).float(), freeze=True)
    else:
        raise NotImplementedError

    # words weighting scheme
    if config.composition == "tf":
        weights_scheme = TfWeights(voc, lambda x: x.to(device))

    elif config.composition == "ones":
        weights_scheme = OnesW(lambda x: x.to(device))

    elif config.composition == "attention":
        weights_scheme = DotProductAttentionWeights(in_s=e_size, key_s=h_size,
                                                    one_hot_input=use_one_hot, to_device=lambda x: x.to(device))
    elif config.composition == "lstm":
        weights_scheme = None

        def composition_layer_factory(w_scheme):
            return BiLSTMEncoding(e_size, e_size//2, lambda x: x.to(device))

    elif config.composition == "tf-idf":
        print("setting up idf embedding...")
        idf_tensor_train = torch.from_numpy(
            np.array([[np.log(n_docs_train + 1) - np.log(df_oc[voc.values[i]] + 1)]
                      for i in range(len(voc))], 'f')
        ).to(device)
        idf_emb_train = nn.Embedding.from_pretrained(idf_tensor_train, freeze=True)
        print("done.")
        weights_scheme = TfidfWeights(TfWeights(voc, lambda x: x.to(device)), idf_emb_train)

    else:
        raise ValueError("{:s} is not a known composition method ".format(config.composition))

    composition_fn = composition_layer_factory(weights_scheme)

    # output layer
    if config.model == "perceptron":
        m_type = Perceptron

    else:
        raise NotImplementedError

    model = BowModel(e_layer, composition_fn, m_type(e_size, h_size))
    # move to device
    model.to(device)
    # loss function
    # (with this sgd should do almost the same as the perceptron algorithm since the loss derivative is 1 at 0)
    per_loss = nn.BCEWithLogitsLoss()
    return model, per_loss


def run_one(train_loader, dev_loader, test_loader, model, loss, measure, device, config, optimizer_type):
    if dev_loader:
        def eval_on_dev():
            test_bow(model, measure, dev_loader, device=device)
            return measure.value()
    else:
        eval_on_dev = None

    opt = optimizer_type(model.parameters(), config.lr)

    train_bow(model, loss, opt, train_loader, config.n_epochs, device=device, epoch_evaluation=eval_on_dev,
              early_stop=config.early_stop)

    model.eval()
    print("evaluating model: ")
    test_bow(model, measure, test_loader, device=device)


def save_model(model, voc, config):
    torch.save(model.state_dict(), os.path.join(config.save_path, f"{config.model_name}.wgt"))
    write_book(voc, config.save_path, config.model_name)


parser = argparse.ArgumentParser(description='train a simple binary classifier on the IMDB dataset')
parser.add_argument("--embedding-type", dest="input", help="type of input (one-hot | glove | learned)",
                    default="learned")
parser.add_argument("--pretrained-we", dest="we_file", help="path to pretrained word-embeddings", default=None)
parser.add_argument("--model-type", dest="model", help="model type (perceptron | linear svm | svm)",
                    default="perceptron")
parser.add_argument("--sentence-composition", dest="composition",
                    help="composition type (ones | tf | attention | tf-idf)", default="tf")
parser.add_argument("--model-name", dest="model_name", help="model name", default=MODEL_NAME)
parser.add_argument("--save-path", dest="save_path", help="path to save directory", default=SAVE_PATH)
parser.add_argument("--lr", dest="lr", help="learning rate", default=LR)
parser.add_argument("--n-epochs", dest="n_epochs", help="learning rate", type=int, default=N_EPOCHS)
parser.add_argument("--batch-size", dest="batch_size", help="batch size", default=B_SIZE)
parser.add_argument("--truncate", dest="truncate", action="store_true", help="truncate corpus", default=TRUNCATE)
parser.add_argument("--no-early-stop", dest="early_stop", action="store_false", help="early stop based on dev stet",
                    default=True)
parser.add_argument("--trunc-size", dest="trunc_size", type=int, help="size of truncated corpus", default=TRUNC_SIZE)
parser.add_argument("--dev-prop", dest="dev_prop", help="dev corpus size in proportion w.r.t. train", default=DEV_PROP)
parser.add_argument("--h-size", dest="h_size", type=int, help="size of hidden layer(s)", default=CUSTOM_E_DIM)

if __name__ == "__main__":

    m_config = parser.parse_args()

    # For reproducibility
    torch.manual_seed(SEED)

    # preprocess (and dev split) corpus
    train_prep, dev_prep, test_prep, voc_train, df_train, gf_train, word_embeddings = preprocess(m_config, SEED)
    print(f"vocabulary size: {len(voc_train):d}")

    # make a pytorch dataloader to feed batches to the train loop
    # wrap the dataset in a pytorch dataset first (and turn inputs to tensors)
    train_dl, dev_dl, test_dl = make_dataloader(voc_train, train_prep, dev_prep, test_prep, m_config)

    # model
    bow_model, bin_loss = build_model(m_config, voc_train, df_train, DEVICE, len(train_prep),
                                      embeddings=word_embeddings)

    accuracy_measure = LinearModelF1(DEVICE)
    # accuracy_measure = LinearModelAccuracy()

    run_one(train_dl, dev_dl, test_dl, bow_model, bin_loss, accuracy_measure, DEVICE, m_config, torch.optim.Adam)

    save_model(bow_model, voc_train, m_config)
    print(accuracy_measure.value())
    if OUTPUT_DEV:
        bow_model.eval()
        print("Analyzing dev set")
        try:
            os.remove("dev_results.tsv")
        except FileNotFoundError:
            pass
        with torch.no_grad():
            for batch in dev_dl:
                x, y = move_to_device(batch, DEVICE, ['x', 'y'])
                logits, composition_weights = bow_model.forward(x, batch)
                for i in range(len(batch['text'])):
                    with open("dev_results.tsv", "a") as dev_res:
                        dev_res.write(batch['cleaned_text'][i])
                        dev_res.write('\t')
                        dev_res.write(str(int(y[i].item())))
                        dev_res.write('\t')
                        dev_res.write(str(int(logits[i].item() > 0)))
                        dev_res.write('\n')



