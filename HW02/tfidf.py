import argparse
import json
from collections import defaultdict
from math import log
import os

from typing import Iterable, Tuple, Dict

from nltk.tokenize import TreebankWordTokenizer
from nltk import FreqDist

kUNK = "<UNK>"

def log10(x):
    return log(x) / log(10)

def lower(str):
    return str.lower()


class TfIdf:
    """Class that builds a vocabulary and then computes tf-idf scores
    given a corpus.

    """

    def __init__(self, vocab_size=10000,
                 tokenize_function=TreebankWordTokenizer().tokenize,
                 normalize_function=lower, unk_cutoff=2):
        self._vocab_size = vocab_size
        self._total_docs = 0

        self._vocab_final = False
        self._vocab = {}
        self._unk_cutoff = unk_cutoff

        self._tokenizer = tokenize_function
        self._normalizer = normalize_function

        # Add your code here!
        self._vocab[kUNK] = 0
        self._train_vocab = defaultdict(int)
        self._doc_freq = defaultdict(int)
        self.num_doc_appears = defaultdict(int)



    def train_seen(self, word: str, count: int=1):
        """Tells the language model that a word has been seen @count times.  This
        will be used to build the final vocabulary.

        word -- The string represenation of the word.  After we
        finalize the vocabulary, we'll be able to create more
        efficient integer representations, but we can't do that just
        yet.

        count -- How many times we've seen this word (by default, this is one).
        """
        if word not in self._train_vocab:
            self._train_vocab[word] = count
        else:
            self._train_vocab[word] += count

        
        assert not self._vocab_final, \
            "Trying to add new words to finalized vocab"

        # Add your code here!

    def add_document(self, text: str):
        """
        Tokenize a piece of text and add the entries to the class's counts.

        text -- The raw string containing a document
        """
        curr_set = set()
        for word in self.tokenize(text):
            self._doc_freq[word] += 1
            if word not in curr_set:
                curr_set.add(word)
                self.num_doc_appears[word] += 1
        self._total_docs += 1

    def tokenize(self, sent: str) -> Iterable[int]:
        """Return a generator over tokens in the sentence; return the vocab
        of a sentence if finalized, otherwise just return the raw string.

        sent -- A string

        """

        # You don't need to modify this code.
        for ii in self._tokenizer(sent):
            if self._vocab_final:
                yield self.vocab_lookup(ii)
            else:
                yield ii

    def doc_tfidf(self, doc: str) -> Dict[Tuple[str, int], float]:
        """Given a document, create a dictionary representation of its tfidf vector

        doc -- raw string of the document"""

        counts = FreqDist(self.tokenize(doc))
        d = {}
        for ii in self._tokenizer(doc):
            ww = self.vocab_lookup(ii)
            d[(ww, ii)] = counts.freq(ww) * self.inv_docfreq(ww)
        return d
                
    def term_freq(self, word: int) -> float:
        """Return the frequence of a word if it's in the vocabulary, zero otherwise.

        word -- The integer lookup of the word.
        """
        #print(self._doc_freq[word])
        if word in self._vocab.values():
            return self._doc_freq[word]/sum(self._doc_freq.values())
        return 0.0

    def inv_docfreq(self, word: int) -> float:
        """Compute the inverse document frequency of a word.  Return 0.0 if
        the word has never been seen.

        Keyword arguments:
        word -- The word to look up the document frequency of a word.

        """
        if word in self._vocab.values():
            return log10(self._total_docs / self.num_doc_appears[word])
        return 0.0

    def vocab_lookup(self, word: str) -> int:
        """
        Given a word, provides a vocabulary integer representation.  Words under the
        cutoff threshold shold have the same value.  All words with counts
        greater than or equal to the cutoff should be unique and consistent.

        This is useful for turning words into features in later homeworks.
        In HW01 we did not specify how to represent each word, here we are using integers

        word -- The word to lookup
        """
        assert self._vocab_final, \
            "Vocab must be finalized before looking up words"

        if word in self._vocab:
            return self._vocab[word]
        else:
            return self._vocab[kUNK]

    def finalize(self):
        """
        Fixes the vocabulary as static, prevents keeping additional vocab from
        being added
        """

        # Add code to generate the vocabulary that the vocab lookup
        # function can use!
        for key in self._train_vocab:
            if self._train_vocab[key] >= self._unk_cutoff:
                self._vocab[key] = self._train_vocab[key]
            else:
                self._vocab[kUNK] += self._train_vocab[key]
    
        self._vocab_final = True

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--root_dir", help="Obituatires",
                           type=str, default='data/',
                           required=False)
    argparser.add_argument("--train_dataset", help="Dataset for training",
                           type=str, default='obits.json',
                           required=False)
    argparser.add_argument("--test_dataset", help="Dataset for test",
                           type=str, default='sparck-jones.txt',
                           required=False)
    argparser.add_argument("--limit", help="Number of training documents",
                           type=int, default=-1, required=False)
    args = argparser.parse_args()

    vocab = TfIdf()

    with open(os.path.join(args.root_dir, args.train_dataset)) as infile:
        data = json.load(infile)["obit"]
        if args.limit > 0:
            data = data[:args.limit]
        for ii in data:
            for word in vocab.tokenize(data[ii]):
                vocab.train_seen(word)
        #print(sorted(((k,v) for k,v in vocab._train_vocab.items()), reverse=False))
        vocab.finalize()
        #print(vocab._vocab)
        for ii in data:
            vocab.add_document(data[ii])
        #print(vocab._doc_freq)

    with open(os.path.join(args.root_dir, args.test_dataset)) as infile:
        #data = json.load(infile)["obit"]
        print(infile)
        vector = vocab.doc_tfidf(infile.read().rstrip())
        for word, tfidf in sorted(vector.items(), key=lambda kv: kv[1], reverse=True)[:50]:
            print("%s:%i\t%f" % (word[1], word[0], tfidf))