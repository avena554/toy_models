import argparse
import os
import sys
import torch
import torch.nn as nn

from utils import compose, read_book
from text_classif_tests import clean_extra_spaces, clean_html, lower, tokenize, TextIndexer, \
    spacy_tokenizer, UNK, CUSTOM_E_DIM, \
    PerceptronOverCombinedWordEmbeddings, Perceptron, StupidSentenceEmbedding


def sub_if_unk(word, voc):
    if word in voc:
        return word
    else:
        return UNK


def sub_unk(inst, voc):
    return {'words': [sub_if_unk(word, voc) for word in inst['words']]}


parser = argparse.ArgumentParser(description='Load and run the stupid classifier on a text read from stdin')
parser.add_argument("--prefix", help="model name", required=True)
parser.add_argument("--path", help="path to the repository containing the weights and voc files", required=True)


args = parser.parse_args()
in_voc = read_book(args.path, args.prefix)

indexer = TextIndexer(in_voc)

spc_tok = spacy_tokenizer()
prepare = compose(indexer, lambda t: sub_unk(t, in_voc), lambda i: tokenize(i, spc_tok),
                  clean_html, clean_extra_spaces, lower)

in_text = sys.stdin.read()
in_text = prepare({"text": in_text})

stupid = PerceptronOverCombinedWordEmbeddings(nn.Embedding(len(in_voc), CUSTOM_E_DIM), StupidSentenceEmbedding(),
                                              Perceptron(CUSTOM_E_DIM))

stupid.load_state_dict(torch.load(f"{os.path.join(args.path, args.prefix)}.wgt"))
stupid.eval()

score = stupid(in_text['x'].unsqueeze(0), torch.ones(1, len(in_text['x'])))

print(torch.sign(score > 0).long().squeeze().item())

