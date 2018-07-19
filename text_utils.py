import re


regex = r"([A-Z](?:\.[A-Z])+\.?|\w+(?:-\w+)*|\$?\d+(?:\.\d+)?%?|\.\.\.|[][.,;\"'?():-_`]|[^\x00-\x7F])"


def TokenizeText(text):
    return re.findall(regex, text.lower())


def ParseNgramsOpts(opts):
    ngrams = [int(g) for g in opts.split(',')]
    ngrams = [g for g in ngrams if (g > 1 and g < 7)]
    return ngrams


def GenerateNgrams(words, ngrams):
    nglist = []
    for ng in ngrams:
        for word in words:
            nglist.extend([word[n:n+ng] for n in range(len(word)-ng+1)])
    return nglist
