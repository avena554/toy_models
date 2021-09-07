import argparse
import os
import sys
import torch
import torch.nn as nn

from utils import read_book
from text_classif_tests import simple_cleanup, as_indices_fn, sub_unknown_fn, spacy_tokenizer, UNK, CUSTOM_E_DIM, \
    PerceptronOverCombinedWordEmbeddings, Perceptron, StupidSentenceEmbedding


parser = argparse.ArgumentParser(description='Load and run the stupid classifier on a text read from stdin')
parser.add_argument("--prefix", help="model name", required=True)
parser.add_argument("--path", help="path to the repository containing the weights and voc files", required=True)


args = parser.parse_args()
in_voc = read_book(args.path, args.prefix)
indexer = as_indices_fn(in_voc)
sub_unknown = sub_unknown_fn(in_voc, unk=UNK)
spc_tok = spacy_tokenizer()

in_text = sys.stdin.read()
x = torch.as_tensor(indexer(sub_unknown(spc_tok(simple_cleanup(in_text))))).unsqueeze(0)

stupid = PerceptronOverCombinedWordEmbeddings(nn.Embedding(len(in_voc), CUSTOM_E_DIM), StupidSentenceEmbedding(),
                                              Perceptron(CUSTOM_E_DIM))

stupid.load_state_dict(torch.load(f"{os.path.join(args.path, args.prefix)}.wgt"))
stupid.eval()

score = stupid(x, torch.ones(1, len(x[0])))

print(torch.sign(score > 0).long().squeeze().item())

