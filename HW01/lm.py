from math import log, exp
from collections import defaultdict
import argparse

from numpy import mean
import re
import nltk
from nltk import FreqDist
from nltk.util import bigrams
from nltk.tokenize import TreebankWordTokenizer
# from nltk.corpus import treebank
# from nltk.tokenize.treebank import TreebankWordDetokenizer

kLM_ORDER = 2
kUNK_CUTOFF = 3
kNEG_INF = -1e6

kSTART = "<s>"
kEND = "</s>"

def lg(x):
    return log(x) / log(2.0)

class BigramLanguageModel:

    def __init__(self, unk_cutoff, jm_lambda=0.6, add_k=0.1,
                 katz_cutoff=5, kn_discount=0.1, kn_concentration=1.0,
                 tokenize_function=TreebankWordTokenizer().tokenize,
                 normalize_function=lambda x: x.lower()):
        self._unk_cutoff = unk_cutoff
        self._jm_lambda = jm_lambda
        self._add_k = add_k
        self._katz_cutoff = katz_cutoff
        self._kn_concentration = kn_concentration
        self._kn_discount = kn_discount
        self._vocab_final = False

        self._tokenizer = tokenize_function
        self._normalizer = normalize_function
        
        # Add your code here!
        self._vocab =  {}
        self._vocab[kSTART] = 0
        self._vocab[kEND] = 0
        self._vocab["<UNK>"] = 0

        self._train_vocab = {}
        self._train_vocab[kSTART] = 0
        self._train_vocab[kEND] = 0
        self._train_vocab["<UNK>"] = 0
        
    def train_seen(self, word, count=1):
        """
        Tells the language model that a word has been seen @count times.  This
        will be used to build the final vocabulary.
        """
        assert not self._vocab_final, \
            "Trying to add new words to finalized vocab"

        # Add your code here!           
        if word in self._vocab:
            self._vocab[word] += count
        else:
            self._vocab[word] = count


    def tokenize(self, sent):
        """
        Returns a generator over tokens in the sentence.  

        You don't need to modify this code.
        """
        for ii in self._tokenizer(sent):
            yield ii
        
    def vocab_lookup(self, word):
        """
        Given a word, provides a vocabulary representation.  Words under the
        cutoff threshold shold have the same value.  All words with counts
        greater than or equal to the cutoff should be unique and consistent.
        """
        assert self._vocab_final, \
            "Vocab must be finalized before looking up words"

        # Add your code here
        if (word in self._vocab and self._vocab[word] >= self._unk_cutoff) or word in [kSTART, kEND]:
            return word
        else:
            return "<UNK>"
        

    def finalize(self):
        """
        Fixes the vocabulary as static, prevents keeping additional vocab from
        being added
        """

        # You probably do not need to modify this code
        self._vocab_final = True

    def tokenize_and_censor(self, sentence):
        """
        Given a sentence, yields a sentence suitable for training or
        testing.  Prefix the sentence with <s>, replace words not in
        the vocabulary with <UNK>, and end the sentence with </s>.

        You should not modify this code.
        """
        yield self.vocab_lookup(kSTART)
        for ii in self._tokenizer(sentence):
            yield self.vocab_lookup(self._normalizer(ii))
        yield self.vocab_lookup(kEND)


    def normalize(self, word):
        """
        Normalize a word

        You should not modify this code.
        """
        return self._normalizer(word)


    def mle(self, context, word):
        """
        Return the log MLE estimate of a word given a context.  If the
        MLE would be negative infinity, use kNEG_INF
        """

        # This initially return 0.0, ignoring the word and context.
        # Modify this code to return the correct value.
        bigram =  context + " " + word if context != "" else word 
        return lg(self._train_vocab[bigram] / self._train_vocab[context]) if (self._train_vocab[context] != 0 and bigram in self._train_vocab) else kNEG_INF
        # return (self._train_vocab[bigram] / self._train_vocab[context]) if (self._train_vocab[context] != 0 and bigram in self._train_vocab) else kNEG_INF


    def vocab_length(self):
        vocab_length = 0
        for key in self._train_vocab:
            if len(key.split()) == 1 and key != "<s>":
                vocab_length += self._train_vocab[key]
        return vocab_length
        
    def laplace(self, context, word):
        """
        Return the log MLE estimate of a word given a context.
        """
        # This initially return 0.0, ignoring the word and context.
        # Modify this code to return the correct value.
        bigram =  context + " " + word if context != "" else word 
        l = self.vocab_length()
        bigram_prob = 0 if bigram not in self._train_vocab else self._train_vocab[bigram]        
        return lg((bigram_prob + 1) / (self._train_vocab[context] + l))

    def jelinek_mercer(self, context, word):
        """
        Return the Jelinek-Mercer log probability estimate of a word
        given a context; interpolates context probability with the
        overall corpus probability.
        """
        # This initially return 0.0, ignoring the word and context.
        # Modify this code to return the correct value.
        bigram =  context + " " + word if context != "" else word 
        bigram_count = 0 if bigram not in self._train_vocab else self._train_vocab[bigram]
        bigram_prob = bigram_count/self._train_vocab[context] if (self._train_vocab[context] != 0) else kNEG_INF
        return lg(self._jm_lambda * bigram_prob + (1 - self._jm_lambda) * (self._train_vocab[word]/self.vocab_length()))

    def kneser_ney(self, context, word):
        """
        Return the log probability of a word given a context given
        Kneser Ney backoff
        """
        # This initially return 0.0, ignoring the word and context.
        # Modify this code to return the correct value.
        return 0.0

    def add_k(self, context, word):
        """
        Additive smoothing, i.e. add_k smoothing, assuming a fixed k
        hyperparameter.
        """
        # This initially return 0.0, ignoring the word and context.
        # Modify this code to return the correct value.
        assert self._add_k > 0.0, "Additive smoothing requires a positive k"

        bigram =  context + " " + word if context != "" else word 
        bigram_prob = 0 if bigram not in self._train_vocab else self._train_vocab[bigram] 
        l = self.vocab_length()       
        return lg((bigram_prob+ self._add_k) / (self._train_vocab[context] + self._add_k * l))

    def add_train(self, sentence):
        """
        Add the counts associated with a sentence.
        """

        # You'll need to complete this function, but here's a line of
        # code that will hopefully get you started.
        #add counts associated with a sentence
        for context, word in bigrams(self.tokenize_and_censor(sentence)):
            # print(context, word)
            

            if word in self._train_vocab:
                self._train_vocab[word] += 1
            else:
                self._train_vocab[word] = 1
            
            if context + " " + word in self._train_vocab:
                self._train_vocab[context + " " + word] += 1
            else:
                self._train_vocab[context + " " + word] = 1
        self._train_vocab[kSTART] += 1

    def perplexity(self, sentence, method):
        """
        Compute the perplexity of a sentence given a estimation method

        You do not need to modify this code.
        """
        return 2.0 ** (-1.0 * mean([method(context, word) for context, word in \
                                    bigrams(self.tokenize_and_censor(sentence))]))
        # return 2.0 ** (-1.0 * mean([method(context, word) for context, word in \
        #                             bigrams(sentence)]))


    def sample(self, method, samples=25):
        """
        Sample words from the language model.
        
        @arg samples The number of samples to return.
        """
        # Modify this code to get extra credit.  This should be
        # written as an iterator.  I.e. yield @samples times followed
        # by a final return, as in the sample code.

        for ii in xrange(samples):
            yield ""
        return

# You do not need to modify the below code, but you may want to during
# your "exploration" of high / low probability sentences.
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--jm_lambda", help="Parameter that controls " + \
                           "interpolation between unigram and bigram",
                           type=float, default=0.6, required=False)
    argparser.add_argument("--add_k", help="Add k value " + \
                           "for pseudocounts",
                           type=float, default=0.1, required=False)
    argparser.add_argument("--unk_cutoff", help="How many times must a word " + \
                           "be seen before it enters the vocabulary",
                           type=int, default=2, required=False)    
    argparser.add_argument("--katz_cutoff", help="Cutoff when to use Katz " + \
                           "backoff",
                           type=float, default=0.0, required=False)
    argparser.add_argument("--lm_type", help="Which smoothing technique to use",
                           type=str, default='mle', required=False)
    argparser.add_argument("--brown_limit", help="How many sentences to add " + \
                           "from Brown",
                           type=int, default=-1, required=False)
    argparser.add_argument("--kn_discount", help="Kneser-Ney discount parameter",
                           type=float, default=0.1, required=False)
    argparser.add_argument("--kn_concentration", help="Kneser-Ney concentration parameter",
                           type=float, default=1.0, required=False)
    argparser.add_argument("--method", help="Which LM method we use",
                           type=str, default='laplace', required=False)
    
    args = argparser.parse_args()    
    lm = BigramLanguageModel(kUNK_CUTOFF, jm_lambda=args.jm_lambda,
                             add_k=args.add_k,
                             katz_cutoff=args.katz_cutoff,
                             kn_concentration=args.kn_concentration,
                             kn_discount=args.kn_discount)

    for ii in nltk.corpus.brown.sents():
        for jj in lm.tokenize(" ".join(ii)):
            lm.train_seen(lm._normalizer(jj))

    print("Done looking at all the words, finalizing vocabulary")
    lm.finalize()

    sentence_count = 0
    for ii in nltk.corpus.brown.sents():
        sentence_count += 1
        lm.add_train(" ".join(ii))

        if args.brown_limit > 0 and sentence_count >= args.brown_limit:
            break

    print("Trained language model with %i sentences from Brown corpus." % sentence_count)
    assert args.method in ['kneser_ney', 'mle', 'add_k', \
                           'jelinek_mercer', 'good_turing', 'laplace'], \
      "Invalid estimation method"

    sent = input()
    
    while sent:
        print("#".join(str(x) for x in lm.tokenize_and_censor(sent)))
        print(lm.perplexity(sent, getattr(lm, args.method)))
        sent = input()

    # sentences = treebank.sents()
    # d = {}
    # for sentence in sentences:
    #     sentence = TreebankWordDetokenizer().detokenize(sentence)
    #     # print(sentence)
    #     s = "#".join(str(x) for x in lm.tokenize_and_censor(sentence))
    #     # print(s)
    #     p = lm.perplexity(sentence, getattr(lm, args.method))
    #     d[sentence] = p
    #     # print(p)
    # sorted_dict = dict(sorted(d.items(), key=lambda item: item[1], reverse = True))
    # print(sorted_dict)