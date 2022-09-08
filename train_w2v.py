from text_classif_tests import simple_cleanup, English
from datasets import load_dataset
import multiprocessing
from gensim.models import Word2Vec, KeyedVectors


spacy_proc = English()
spacy_proc.add_pipe('sentencizer')
imdb = load_dataset("imdb")
train = imdb['train']
test = imdb['test']

docs = []
sents = []
for ds in [train, test]:
    for instance in ds['text']:
        docs.append(simple_cleanup(instance))

for doc in spacy_proc.pipe(docs, batch_size=1, n_process=-1):
    sents.extend([[str(t) for t in s] for s in doc.sents])

print(len(docs))
print(sents[0])

cores = multiprocessing.cpu_count()
w2v_model = Word2Vec(sentences=sents, min_count=3,
                     window=5,
                     size=200,
                     sample=6e-5,
                     alpha=0.03,
                     min_alpha=0.0007,
                     negative=20,
                     workers=cores-1)

vectors = w2v_model.wv
vectors.save_word2vec_format("/part/02/data/venanta/embeddings/w2v_200.txt", binary=False)
#w2v_model.build_vocab(sents, progress_per=10000)
#w2v_model.train(sents, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)




