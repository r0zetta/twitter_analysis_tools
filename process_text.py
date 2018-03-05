# -*- coding: utf-8 -*-
import re
import spacy
from nltk.stem import PorterStemmer

def get_tweet_tags(tweet):
    preprocessed = preprocess_text(tweet)
    if preprocessed is not None:
        return process_sentence(preprocessed)

def preprocess_text(text):
    valid = u"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ#@'…… "
    url_match = u"(https?:\/\/[0-9a-zA-Z\-\_]+\.[\-\_0-9a-zA-Z]+\.?[0-9a-zA-Z\-\_]*\/?.*)"
    name_match = u"\@[\_0-9a-zA-Z]+\:?"
    text = re.sub(url_match, u"", text)
    text = re.sub(name_match, u"", text)
    text = re.sub(u"\&amp\;?", u"", text)
    text = re.sub(u"[\:\.]{1,}$", u"", text)
    text = re.sub(u"^RT\:?", u"", text)
    text = re.sub(u"/", u" ", text)
    text = re.sub(u"-", u" ", text)
    text = re.sub(u"\w*[\…]", u"", text)
    text = u''.join(x for x in text if x in valid)
    text = text.strip()
    if len(text.split()) > 5:
        return text

# Tokenize sentence into words
def tokenize_sentence(text, stopwords=None):
    words = re.split(r'(\s+)', text)
    if len(words) < 1:
        return
    tokens = []
    for w in words:
        if w is not None:
            w = w.strip()
            w = w.lower()
            if w.isspace() or w == "\n" or w == "\r":
                w = None
            if w is not None and "http" in w:
                w = None
            if w is not None and len(w) < 1:
                w = None
            if w is not None and u"…" in w:
                w = None
            if w is not None:
                tokens.append(w)
    if len(tokens) < 1:
        return
# Remove stopwords and other undesirable tokens
    cleaned = []
    for token in tokens:
        if len(token) > 0:
            if stopwords is not None:
                for s in stopwords:
                    if token == s:
                        token = None
            if token is not None:
                if re.search(".+…$", token):
                    token = None
            if token is not None:
                if token == "#":
                    token = None
            if token is not None:
                cleaned.append(token)
    if len(cleaned) < 1:
        return
    return cleaned



def get_tokens(doc, stemmer):
    ret = []
    for token in doc:
        lemma = token.lemma_
        pos = token.pos_
        if pos in ["VERB", "ADJ", "ADV", "NOUN"]:
            if lemma.lower() not in ["are", "do", "'s", "be", "is", "https//", "#", "-pron-", "so", "as", "that", "not", "who", "which", "thing", "even", "said", "says", "say", "keep", "like", "will", "have", "what", "can", "how", "get", "there", "would", "when", "then", "here", "other", "know", "let", "all"]:
                if len(lemma) > 1:
                    stem = stemmer.stem(lemma)
                    ret.append(stem)
    return ret

def get_labels(doc):
    labels = []
    for entity in doc.ents:
        label = entity.label_
        text = entity.text
        if label in ["ORG", "GPE", "PERSON", "NORP"]:
            if len(text) > 1:
                labels.append(text)
    return labels

def get_hashtags(sentence):
    ret = []
    words = re.split(r'(\s+)', sentence)
    for w in words:
        if re.search("^\#[a-zA-Z0-9]+$", w):
            if w not in ret:
                ret.append(w)
    return ret

def process_sentence_nlp(sentence, nlp, stemmer):
    tags = []
    doc = nlp(sentence)
    # get tags using spacy
    tokens = get_tokens(doc, stemmer)
    for t in tokens:
        if t not in tags:
            tags.append(t)
    labels = get_labels(doc)
    for l in labels:
        if l not in tags:
            tags.append(l)
    hashtags = get_hashtags(sentence)
    for h in hashtags:
        if h not in tags:
            tags.append(h)
    # lowercase and remove duplicates
    cons = []
    for t in tags:
        t = t.lower()
        t = t.strip()
        if t not in cons:
            cons.append(t)
    return cons

def is_bot_name(name):
    ret = True
    if re.search("^([A-Z]?[a-z]{1,})?[\_]?([A-Z]?[a-z]{1,})?[\_]?[0-9]{,9}$", name):
        ret = False
    if re.search("^[\_]{,3}[A-Z]{2,}[\_]{,3}$", name):
        ret = False
    if re.search("^[A-Z]{2}[a-z]{2,}$", name):
        ret = False
    if re.search("^([A-Z][a-z]{1,}){3}[0-9]?$", name):
        ret = False
    if re.search("^[A-Z]{1,}[a-z]{1,}[A-Z]{1,}$", name):
        ret = False
    if re.search("^[A-Z]{1,}[a-z]{1,}$", name):
        ret = False
    if re.search("^([A-Z]?[a-z]{1,}[\_]{1,}){1,}[A-Z]?[a-z]{1,}$", name):
        ret = False
    if re.search("^[A-Z]{1,}[a-z]{1,}[\_][A-Z][\_][A-Z]{1,}[a-z]{1,}$", name):
        ret = False
    if re.search("^[a-z]{1,}[A-Z][a-z]{1,}[A-Z][a-z]{1,}$", name):
        ret = False
    if re.search("^[A-Z][a-z]{1,}[A-Z][a-z]{1,}[A-Z]{1,}$", name):
        ret = False
    if re.search("^([A-Z][\_]){1,}[A-Z][a-z]{1,}$", name):
        ret = False
    if re.search("^[\_][A-Z][a-z]{1,}[\_][A-Z][a-z]{1,}[\_]?$", name):
        ret = False
    if re.search("^[A-Z][a-z]{1,}[\_][A-Z][\_][A-Z]$", name):
        ret = False
    if re.search("^[A-Z][a-z]{2,}[0-9][A-Z][a-z]{2,}$", name):
        ret = False
    if re.search("^[A-Z]{1,}[0-9]?$", name):
        ret = False
    if re.search("^[A-Z][a-z]{1,}[\_][A-Z]$", name):
        ret = False
    if re.search("^[A-Z][a-z]{1,}[A-Z]{2}[a-z]{1,}$", name):
        ret = False
    if re.search("^[\_]{1,}[a-z]{2,}[\_]{1,}$", name):
        ret = False
    if re.search("^[A-Z][a-z]{2,}[\_][A-Z][a-z]{2,}[\_][A-Z]$", name):
        ret = False
    if re.search("^[A-Z]?[a-z]{2,}[0-9]{2}[\_]?[A-Z]?[a-z]{2,}$", name):
        ret = False
    if re.search("^[A-Z][a-z]{2,}[A-Z]{1,}[0-9]{,2}$", name):
        ret = False
    if re.search("^[\_][A-Z][a-z]{2,}[A-Z][a-z]{2,}[\_]$", name):
        ret = False
    if re.search("^([A-Z][a-z]{1,}){2,}$", name):
        ret = False
    if re.search("^[A-Z][a-z]{2,}[\_][A-Z]{2}$", name):
        ret = False
    if re.search("^[a-z]{3,}[0-9][a-z]{3,}$", name):
        ret = False
    if re.search("^[a-z]{4,}[A-Z]{1,}$", name):
        ret = False
    if re.search("^[A-Z][a-z]{3,}[A-Z][0-9]{,9}$", name):
        ret = False
    if re.search("^[A-Z]{2,}[\_][A-Z][a-z]{3,}$", name):
        ret = False
    if re.search("^[A-Z][a-z]{3,}[A-Z]{1,3}[a-z]{3,}$", name):
        ret = False
    if re.search("^[A-Z]{3,}[a-z]{3,}[0-9]?$", name):
        ret = False
    if re.search("^[A-Z]?[a-z]{3,}[\_]+$", name):
        ret = False
    if re.search("^[A-Z][a-z]{3,}[\_][a-z]{3,}[\_][A-Za-z]{1,}$", name):
        ret = False
    if re.search("^[A-Z]{2,}[a-z]{3,}[A-Z][a-z]{3,}$", name):
        ret = False
    if re.search("^[A-Z][a-z]{2,}[A-Z][a-z]{3,}[\_]?[A-Z]{1,}$", name):
        ret = False
    if re.search("^[A-Z]{4,}[0-9]{2,9}$", name):
        ret = False
    if re.search("^[A-Z]{1,2}[a-z]{3,}[A-Z]{1,2}[a-z]{3,}[0-9]{1,9}$", name):
        ret = False
    if re.search("^[A-Z]+[a-z]{3,}[0-9]{1,9}$", name):
        ret = False
    if re.search("^([A-Z]?[a-z]{2,})+[0-9]{1,9}$", name):
        ret = False
    if re.search("^([A-Z]?[a-z]{2,})+\_?[a-z]+$", name):
        ret = False
    return ret
