import argparse
import os
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from utils import read_book
from text_classif_tests import simple_cleanup, as_indices_fn, sub_unknown_fn, spacy_tokenizer, UNK, CUSTOM_E_DIM, \
    BowModel, Perceptron, StupidSentenceEmbedding, \
    AttentionSentenceEmbedding, DotProductAttentionSentenceEmbedding


parser = argparse.ArgumentParser(description='Load and run the stupid classifier on a text read from stdin')
parser.add_argument("--prefix", help="model name", required=True)
parser.add_argument("--path", help="path to the repository containing the weights and voc files", required=True)


args = parser.parse_args()
in_voc = read_book(args.path, args.prefix)
indexer = as_indices_fn(in_voc)
sub_unknown = sub_unknown_fn(in_voc, unk=UNK)
spc_tok = spacy_tokenizer()

in_text = sys.stdin.read()
words = sub_unknown(spc_tok(simple_cleanup(in_text)))
x = torch.as_tensor(indexer(words)).unsqueeze(0)


class AttentionHook:

    def __init__(self):
        self.weights = None

    def __call__(self, nn_inst, scores, weights):
        self.weights = weights


attention_layer = DotProductAttentionSentenceEmbedding(CUSTOM_E_DIM)
ww = AttentionHook()
attention_layer.norm.register_forward_hook(ww)

stupid = BowModel(nn.Embedding(len(in_voc), CUSTOM_E_DIM),
                  attention_layer,
                  Perceptron(CUSTOM_E_DIM))

stupid.load_state_dict(torch.load(f"{os.path.join(args.path, args.prefix)}.wgt"))
stupid.eval()

score = stupid(x, torch.ones(1, len(x[0])))
probas = ww.weights.squeeze().detach().numpy()
print(words)
print(probas)
fig = plt.figure()
ax = fig.add_axes([0.2, 0.2, 0.8, 0.8])
ax.bar(words[0:70], probas[0:70])
plt.xticks(rotation=90)
plt.show()

print(torch.sign(score > 0).long().squeeze().item())

