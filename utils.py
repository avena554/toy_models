import os
from collections import defaultdict


class OccurrencesCount:

    def __init__(self):
        self._occurrences = defaultdict(int)
        self._total = 0

    def incr(self, token):
        self._occurrences[token] += 1
        self._total += 1

    def len(self):
        return len(self._occurrences)

    def total(self):
        return self._total

    def __getitem__(self, token):
        return self._occurrences[token]

    def __iter__(self):
        return iter(self._occurrences)

    def __contains__(self, token):
        return token in self._occurrences


class Book:

    def __init__(self):
        self._indices = dict()
        self.values = []
        self._size = 0

    @staticmethod
    def from_dict(d):
        book = Book()
        book._indices = dict(d)
        book._fill_inverse()
        book._size = len(d)
        return book

    def _fill_inverse(self):
        self.words = ['']*len(self)
        for (w, i) in self._indices.items():
            self.words[i] = w

    def _add_wo_check(self, k):
        self._indices[k] = self._size
        self._size += 1
        self.values.append(k)

    def add(self, k):
        if k not in self:
            self._add_wo_check(k)

    def __contains__(self, k):
        return k in self._indices

    def __len__(self):
        return self._size

    def __getitem__(self, k):
        return self._indices[k]

    def __iter__(self):
        return iter(self._indices)

    def items(self):
        return self._indices.items()

    def __str__(self):
        return self._indices.__str__()


def write_book(book, path, name, repr_key=lambda x: x):
    base = os.path.join(path, name)
    with open(f"{base}.voc", "w", encoding="UTF-8") as fdesc:
        for key in book:
            fdesc.write(f"{repr_key(key)}\n")
    with open(f"{base}.idx", "w", encoding="UTF-8") as fdesc:
        for key in book:
            fdesc.write(f"{book[key]:d}\n")

    fdesc.close()


def read_book(path, name, read_key=lambda x: x):
    d = dict()
    base = os.path.join(path, name)
    with open(f"{base}.voc", "r", encoding="UTF-8") as f_voc:
        with open(f"{base}.idx", "r", encoding="UTF-8") as f_idx:
            for key_str, index in zip(f_voc, f_idx):
                d[read_key(key_str)[:-1]] = int(index[:-1])

    return Book.from_dict(d)


def compose(*fns):
    def composition(arg):
        for fn in reversed(fns):
            arg = fn(arg)

        return arg
    return composition
