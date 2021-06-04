"""
This file is available at http://www.cs.cmu.edu/~dhuggins/Projects/pyphone/sphinx/arpalm.py
For DSTC4 we have done some modifications to adapt it to our needs. all changes are commented
starting with the word DSTC4
"""

# Copyright (c) 2006 Carnegie Mellon University
#
# You may copy and modify this freely under the same terms as
# Sphinx-III

"""Read ARPA-format language models.

This module provides a class for reading, writing, and using ARPA
format statistical language model files.
"""

"""
__author__ = "David Huggins-Daines <dhuggins@cs.cmu.edu>"
__version__ = "$Revision: 7505 $"
"""


import numpy
import gzip
import re
import codecs  # DSTC5: Include to allow reading files with UTF-8

LOG10TOLOG = numpy.log(10)

class ArpaLM(object):
    "Class for reading ARPA-format language models"

    def __init__(self, path=None):
        """
        Initialize an ArpaLM object.

        @param path: Path to an ARPA format file to (optionally) load
                     language model from.  This file can be
                     gzip-compressed if you like.
        @type path: string
        """
        if path != None:
            self.read(path)

    def read(self, path):
        """
        Load an ARPA format language model from a file in its entirety.

        @param path: Path to an ARPA format file to (optionally) load
                     language model from.  This file can be
                     gzip-compressed if you like.
        @type path: string
        """
        # print('Loading LM: '+ path)
        if path.find('.gz') != -1:  # DSTC4: Detection of the type of file, allows for gz or text format
            zf = gzip.open(path, 'rb')
            reader = codecs.getreader("utf-8")
            fh = reader(zf)
        else:
            fh = codecs.open(path, 'r', 'utf-8')  # DSTC5: Allows reading utf-8 files

        # Skip header
        while True:
            spam = fh.readline().rstrip()
            if spam == "\\data\\":
                break

        # Get N-gram counts
        self.ng_counts = {}
        r = re.compile(r"ngram (\d+)=(\d+)")
        while True:
            spam = fh.readline().rstrip()
            if spam == "":
                break
            m = r.match(spam)
            if m != None:
                n, c = map(int, m.groups())
                self.ng_counts[n] = c

        # Word and N-Gram to ID mapping
        self.ngmap = []
        # Create probability/backoff arrays
        self.n = max(self.ng_counts.keys())
        self.ngrams = []
        for n in range(1,self.n+1):
            vals = numpy.zeros((self.ng_counts[n]+1,2),'d')  # DSTC4: Add 1 more position for the unk case just in case it does not exists
            self.ngrams.append(vals)
            self.ngmap.append({})

        # Read unigrams and create word id list
        spam = fh.readline().rstrip()
        if spam != "\\1-grams:":
            raise Exception("1-grams marker not found")
        # ID to word mapping
        self.widmap = []
        wordid = 0
        while True:
            spam = fh.readline().rstrip()
            if spam == "":
                break

            # DSTC4: Allows ARPA models without Backoff weights for each n-gram (e.g. typical configuration in srilm)
            try:
                p,w,b = spam.split()
            except:
                p,w = spam.split()
                b = 0.0

            self.ngmap[0][w] = wordid
            self.widmap.append(w)
            self.ngrams[0][wordid,:] = (float(p) * LOG10TOLOG,
                                        float(b) * LOG10TOLOG)
            wordid = wordid + 1

        # DSTC4: In case the unk does not exists in the model, it gives the zero-gram probability
        if '<unk>' not in self.ngmap[0]:
            self.ngmap[0]['<unk>'] = wordid
            self.widmap.append('<unk>')
            self.ngrams[0][wordid,:] = (float(numpy.log10(1.0/wordid)) * LOG10TOLOG,
                                        float(0) * LOG10TOLOG)

        # Read N-grams
        r = re.compile(r"\\(\d+)-grams:")
        ngramid = 0
        # Successor list map
        self.succmap = {}
        while True:
            spam = fh.readline().rstrip()
            if spam == "":
                continue
            if spam == "\\end\\":
                break
            m = r.match(spam)
            if m != None:
                n = int(m.group(1))
                ngramid = 0
            else:
                spam = spam.split()
                p = float(spam[0]) * LOG10TOLOG
                if n == self.n:
                    ng = tuple(spam[1:])
                    b = 0.0
                else:
                    # DSTC4: Allows ARPA models without Backoff weights for each n-gram (e.g. typical configuration in srilm)
                    if len(spam) == n + 2:  # the n-gram has a backoff value
                        ng = tuple(spam[1:-1])
                        try:
                            b = float(spam[-1]) * LOG10TOLOG
                        except:
                            b = 0.0
                    else:
                        ng = tuple(spam[1:])
                        b = 0.0
                # N-Gram info
                self.ngrams[n-1][ngramid,:] = p, b
                self.ngmap[n-1][ng] = ngramid

                # Successor list for N-1-Gram
                mgram = tuple(ng[:-1])
                if mgram not in self.succmap:
                    self.succmap[mgram] = []
                self.succmap[mgram].append(ng[-1])
                ngramid = ngramid + 1

    def save(self, path):
        """
        Save an ARPA format language model to a file.

        @param path: Path to save the file to.  If this ends in '.gz',
                     the file contents will be gzip-compressed.
        @type path: string
        """
        if path.endswith('.gz'):
            fh = gzip.open(path, 'w')
        else:
            fh = open(path, 'w')
        fh.write("# Written by arpalm.py\n")
        fh.write("\\data\\\n")
        for n in range(1, self.n+1):
            fh.write("ngram %d=%d\n" % (n, self.ng_counts[n]))
        for n in range(1, self.n+1):
            fh.write("\n\\%d-grams:\n" % n)
            ngrams = self.ngmap[n-1].keys()
            ngrams.sort()
            if '<unk>' in self.ngmap[n-1]:
                ngid = self.ngmap[n-1]['<unk>']
                score, bowt = self.ngrams[n-1][ngid]
                if n == self.n:
                    fh.write("%.4f <unk>\n" % (score))
                else:
                    fh.write("%.4f <unk>\t%.4f\n" % (score,bowt))
            for g in ngrams:
                if g == '<unk>':
                    continue
                ngid = self.ngmap[n-1][g]
                score, bowt = self.ngrams[n-1][ngid]
                if n > 1:
                    g = " ".join(g)
                if n == self.n:
                    fh.write("%.4f %s\n" % (score, g))
                else:
                    fh.write("%.4f %s\t%.4f\n" % (score, g, bowt))
        fh.write("\n\\end\\\n")
        fh.close()

    def successors(self, *syms):
        """
        Return all successor words for an M-Gram

        @return: The list of end words for all (M+1)-Grams begining
                 with the words given.
        @rtype: [string]
        """
        try:
            return self.succmap[syms]
        except:
            return []

    def score(self, *syms):
        """
        Return the language model log-probability for an N-Gram

        @return: The log probability for the N-Gram consisting of the
                 words given, in base e (natural log).
        @rtype: float
        """
        # It makes the most sense to do this recursively
        verboseLevel = 0

        # These comprobations allow to make the call for each ngram order from outside the module and from the recursion
        if type(syms) is tuple and type(syms[0]) is tuple:  # The first call
            symsT = syms[0]
            n = len(symsT)
        elif type(syms) is tuple:
            symsT = syms
            n = len(symsT)
        else:
            symsT = syms
            n = 1

        if n == 1:
            if symsT[0] in self.ngmap[0]:
                # 1-Gram exists, just return its probability
                v = self.ngrams[0][self.ngmap[0][symsT[0]]][0]
                if verboseLevel > 1 : print("%s %d-gram %f" %(symsT[0], n, v))  # DSTC4 : Make the stdout less verbose
                return v
            else:
                # Use <unk>
                v = self.ngrams[0][self.ngmap[0]['<unk>']][0]
                if verboseLevel > 1 : print("%s %d-gram %f" %('unk', n, v)) # DSTC4 : Make the stdout less verbose
                return v
        else:
            if symsT in self.ngmap[n-1]:
                # N-Gram exists, just return its probability
                v = self.ngrams[n-1][self.ngmap[n-1][symsT]][0]
                if verboseLevel > 1 : print("%s %d-gram %f" %(symsT, n, v)) # DSTC4 : Make the stdout less verbose
                return v
            else:
                # Backoff: alpha(history) * probability (N-1-Gram)
                hist = tuple(symsT[0:-1])
                symsT = symsT[1:]
                # Treat unigram histories a bit specially
                if n == 2:
                    hist = hist[0]
                    # Back off to <unk> if word doesn't exist
                    if not hist in self.ngmap[0]:
                        hist = '<unk>'
                if hist in self.ngmap[n-2]:
                    # Try to use the history if it exists
                    bowt = self.ngrams[n-2][self.ngmap[n-2][hist]][1]
                    v = bowt + self.score(*symsT)
                    if verboseLevel > 1 : print("%s %d-gram %f" %(symsT, n, v)) # DSTC4 : Make the stdout less verbose
                    return v
                else:
                    # Otherwise back off some more
                    v = self.score(*symsT)
                    if verboseLevel > 1 : print("%s %d-gram %f" %(symsT, n, v)) # DSTC4 : Make the stdout less verbose
                    return v

    def adapt_rescale(self, unigram, vocab=None):
        """Update unigram probabilities with unigram (assumed to be in
        linear domain), then rescale N-grams ending with the same word
        by the corresponding factors.  If unigram is not the same size
        as the original vocabulary, you must pass vocab, which is a
        list of the words in unigram, in the same order as their
        probabilities are listed in unigram."""
        if vocab:
            # Construct a temporary list mapping for the unigrams
            vmap = map(lambda w: self.ngmap[0][w], vocab)
            # Get the original unigrams
            og = numpy.exp(self.ngrams[0][:,0].take(vmap))
            # Compute the individual scaling factors
            ascale = unigram * og.sum() / og
            # Put back the normalized version of unigram
            self.ngrams[0][:,0].put(numpy.log(unigram * og.sum()), vmap)
            # Now reconstruct vocab as a dictionary mapping words to
            # scaling factors
            vv = {}
            for i, w in enumerate(vocab):
                vv[w] = i
            vocab = vv
        else:
            ascale = unigram / numpy.exp(self.ngrams[0][:,0])
            self.ngrams[0][:,0] = numpy.log(unigram)

        for n in range(1, self.n):
            # Total discounted probabilities for each history
            tprob = numpy.zeros(self.ngrams[n-1].shape[0], 'd')
            # Rescaled total probabilities
            newtprob = numpy.zeros(self.ngrams[n-1].shape[0], 'd')
            # For each N-gram, accumulate and rescale
            for ng,idx in self.ngmap[n].iteritems():
                h = ng[0:-1]
                if n == 1: # Quirk of unigrams
                    h = h[0]
                w = ng[-1]
                prob = numpy.exp(self.ngrams[n][idx,0])
                tprob[self.ngmap[n-1][h]] += prob
                if vocab == None or w in vocab:
                    prob = prob * ascale[vocab[w]]
                newtprob[self.ngmap[n-1][h]] += prob
                self.ngrams[n][idx,0] = numpy.log(prob)
            # Now renormalize everything
            norm = tprob / newtprob
            for ng,idx in self.ngmap[n].iteritems():
                h = ng[0:-1]
                if n == 1: # Quirk of unigrams
                    h = h[0]
                w = ng[-1]
                prob = numpy.exp(self.ngrams[n][idx,0])
                self.ngrams[n][idx,0] = numpy.log(prob * norm[self.ngmap[n-1][h]])
