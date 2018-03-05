# -*- coding: utf-8 -*-
from nltk.stem import PorterStemmer
from six.moves import cPickle
from collections import Counter
from itertools import combinations
from sets import Set
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from nltk.cluster.util import cosine_distance
from operator import itemgetter
from difflib import SequenceMatcher
from random import randint
import numpy as np
import pickle
import math
import spacy
import io
import os
import sys
import io
import re
import json


def print_progress(current, maximum):
    sys.stdout.write("\r")
    sys.stdout.flush()
    sys.stdout.write(str(current) + "/" + str(maximum))
    sys.stdout.flush()

def save_json(variable, filename):
    with io.open(filename, "w", encoding="utf-8") as f:
        f.write(unicode(json.dumps(variable, indent=4, ensure_ascii=False)))

def load_json(filename):
    ret = None
    if os.path.exists(filename):
        try:
            with io.open(filename, "r", encoding="utf-8") as f:
                ret = json.load(f)
        except:
            pass
    return ret

def try_load_or_process(filename, processor_fn, function_arg):
    load_fn = None
    save_fn = None
    if filename.endswith("json"):
        load_fn = load_json
        save_fn = save_json
    else:
        load_fn = load_bin
        save_fn = save_bin
    if os.path.exists(filename):
        print("Loading " + filename)
        return load_fn(filename)
    else:
        ret = processor_fn(function_arg)
        print("Saving " + filename)
        save_fn(ret, filename)
        return ret

def load_raw_data(raw_input_file):
    ret = []
    print("Loading raw data from: " + raw_input_file)
    if os.path.exists(raw_input_file):
        with io.open(raw_input_file, 'r', encoding="utf-8") as f:
            ret = f.readlines()
    return ret

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

def process_raw_data(input_file):
    ret = []
    lines = load_raw_data(input_file)
    if lines is not None and len(lines) > 0:
        num_lines = len(lines)
        for count, text in enumerate(lines):
            print_progress(count, num_lines)
            processed = preprocess_text(text)
            if processed is not None:
                if processed not in ret:
                    ret.append(processed)
    return ret

def get_tokens(doc):
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
    global g_labels
    labels = []
    for entity in doc.ents:
        label = entity.label_
        text = entity.text
        if label in ["ORG", "GPE", "PERSON", "NORP"]:
            if len(text) > 1:
                labels.append(text)
                g_labels[text] += 1
    return labels

def get_hashtags(sentence):
    ret = []
    words = re.split(r'(\s+)', sentence)
    for w in words:
        if re.search("^\#[a-zA-Z0-9]+$", w):
            if w not in ret:
                ret.append(w)
    return ret

def process_sentence(sentence):
    doc = nlp(sentence)
    tags = []
    # get tags using spacy
    tokens = get_tokens(doc)
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

def process_sentences(processed):
    tag_map = {}
    print("Processing sentences.")
    num_sentences = len(processed)
    for count, sentence in enumerate(processed):
        print_progress(count + 1, num_sentences)
        tag_map[sentence] = process_sentence(sentence)
    filename = os.path.join(save_dir, "g_labels.json")
    save_json(g_labels, filename)
    return tag_map

def get_freq_dist(tag_map):
    print("Creating frequency distribution.")
    dist = Counter()
    count = 1
    tag_map_size = len(tag_map)
    for tweet, tags in tag_map.iteritems():
        print_progress(count, tag_map_size)
        count += 1
        for t in tags:
            if t not in ["just", "think", "how", "need", "only", "all", "still", "even", "why", "look", "let", "most", "way", "more", "mean", "new", "must", "talk", "try", "back", "have", "seem", "will", "see", "use", "tell", "would", "should", "could", "can", "go", "are", "do", "'s", "be", "make", "want", "know", "come", "is", "https//", "#", "-pron-", "when", "here", "say", "there", "also", "quite", "so", "get", "perhaps", "as", "that", "now", "not", "then", "who", "very", "which", "then", "thing", "what", "take", "give", "show", "really", "keep", "other", "people", "man", ]:
                dist[t] += 1
    print
    print("Total unique tags: " + str(len(dist)))
    return dist

def vectorize_item(tags, vocab):
    row = []
    for word in vocab:
        if word in tags:
            row.append(1)
        else:
            row.append(0)
    return row

def vectorize_list(tag_map, vocab, dataset):
    vectors = []
    max_s = len(dataset)
    for index, tweet in enumerate(dataset):
        print_progress(index, max_s)
        tags = tag_map[tweet]
        row = vectorize_item(tags, vocab)
        vectors.append(row)
    return vectors

def make_clusters(X, tweets, pca_k, num_k, rand_seed):
    print("Random seed: " + str(rand_seed))
    print("PCA components: " + str(pca_k))
    print("Current k=" + str(num_k))
    print("Computing PCA")
    pca = PCA(n_components=pca_k,
              random_state=rand_seed,
              whiten=True)
    X_pca = pca.fit_transform(X)

    print("Computing k means")
    model = KMeans(n_clusters=num_k,
                   init='random',
                   max_iter=100,
                   n_init=10,
                   random_state=rand_seed,
                   verbose=0)
    model.fit(X_pca)

    print("Predicting")
    clusters = {}
    for index, Y in enumerate(X_pca):
        prediction = model.predict(Y.reshape(1,-1))
        category = "cluster" + "%02d" % prediction[0]
        if category not in clusters:
            clusters[category] = []
        clusters[category].append(tweets[index])
    return clusters, model, pca

def cluster(tag_map, tweets):
    print("Calculating frequency distribution")
    filename = os.path.join(save_dir, "freq_dist.json")
    dist = Counter(try_load_or_process(filename, get_freq_dist, tag_map))

    min_hits = 5
    index = 0
    num_features = 0
    for x, c in dist.most_common():
        if c < min_hits:
            num_features = index
            break
        index += 1

    print("Frequency dist below " + str(min_hits) + " hits was at: " + str(num_features))

    print("Creating vocab list and mapping")
    vocab_list = [item for item, count in dist.most_common()]
    print(str(len(vocab_list)) + " unique tokens in text.")
    vocab_map = {item:index for index, item in enumerate(vocab_list)}
    common_vocab = [item for item, count in dist.most_common(num_features)]

    print("Saving vocab map")
    filename = os.path.join(save_dir, "vocab_map.json")
    save_json(vocab_map, filename)
    filename = os.path.join(save_dir, "common_vocab.json")
    save_json(common_vocab, filename)

    print("Creating vectors")
    vectors = np.array(vectorize_list(tag_map, common_vocab, tweets))

    pca_k = 2
    num_k = 130
    rand_seed = 9
    clusters, model, pca = make_clusters(vectors, tweets, pca_k, num_k, rand_seed)
    cluster_lengths = []
    current_max = 0
    for c, t in clusters.iteritems():
        if len(t) > current_max:
            current_max = len(t)
        cluster_lengths.append(len(t))
    print("Max cluster length: " + str(current_max))
    filename = os.path.join(save_dir, "cluster_lengths.json")
    save_json(cluster_lengths, filename)
    filename = os.path.join(save_dir, "k_means_model.sav")
    pickle.dump(model, open(filename, "wb"))
    filename = os.path.join(save_dir, "pca_model.sav")
    pickle.dump(pca, open(filename, "wb"))
    return clusters

def pagerank(A, eps=0.0001, d=0.85):
    P = np.ones(len(A)) / len(A)
    while True:
        new_P = np.ones(len(A)) * (1 - d) / len(A) + d * A.T.dot(P)
        delta = abs((new_P - P).sum())
        if delta <= eps:
            return new_P
        P = new_P
 
def sentence_similarity(sent1, sent2):
    all_words = list(set(sent1 + sent2))
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    for w in sent1:
        vector1[all_words.index(w)] += 1
 
    for w in sent2:
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences):
    S = np.zeros((len(sentences), len(sentences)))
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 != idx2:
                S[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2])
 
    for idx in range(len(S)):
        if S[idx].sum() != 0:
            S[idx] /= S[idx].sum()
    return S

def textrank(tag_map, sentences, top_n=5):
    if len(sentences) < 1:
        return []
    if len(sentences) <= top_n:
        return sentences
    tokenized = []
    for s in sentences:
        tokenized.append(tag_map[s])
    S = build_similarity_matrix(tokenized)
    if not np.all(S):
        sentence_ranks = pagerank(S)
        ranked_sentence_indexes = [item[0] for item in sorted(enumerate(sentence_ranks), key=lambda item: -item[1])]
        selected_sentences = sorted(ranked_sentence_indexes[:top_n])
        summary = []
        for index in selected_sentences:
            summary.append(sentences[index])
        return summary
    else:
        return []

def summarize_cluster_map(tag_map, cluster_map):
    summaries = {}
    batch_len = 200
    filename = os.path.join(save_dir, "temp_summaries.json")
    max_s = len(cluster_map)
    count = 0
    for cluster, tweets in cluster_map.iteritems():
        count += 1
        print_progress(count, max_s)
        if len(tweets) > 2000:
            batched_sum = []
            batches = (tweets[i:i+batch_len] for i in range(0, len(tweets), batch_len))
            for b in batches:
                batched_sum += textrank(tag_map, b)
            summaries[cluster] = textrank(tag_map, batched_sum)
        else:
            summaries[cluster] = textrank(tag_map, tweets)
        save_json(summaries, filename)
    return summaries

def test_predict():
    filename = os.path.join(save_dir, "data.json")
    processed = try_load_or_process(filename, process_raw_data, raw_input_file)
    print("Testing predict function.")
    for n in range(20):
        test = processed[randint(0,len(processed)-1)]
        print test
        category = predict_tweet(test)
        print category

def predict_tweet(tweet):
    tweet = preprocess_text(tweet)
    if tweet is not None:
        tags = process_sentence(tweet)
        if tags is not None:
            filename = os.path.join(save_dir, "common_vocab.json")
            common_vocab = load_json(filename)
            row = vectorize_item(tags, common_vocab)

            Y = np.array(row)
            pca = PCA()
            filename = os.path.join(save_dir, "pca_model.sav")
            pca = pickle.load(open(filename, "rb"))
            Y = pca.transform(Y.reshape(1,-1))

            model = KMeans()
            filename = os.path.join(save_dir, "k_means_model.sav")
            model = pickle.load(open(filename, "rb"))
            prediction = model.predict(Y)
            category = "k_means_cluster" + "%02d" % prediction[0]
            return category

def counter_cosine_similarity(c1, c2):
    terms = set(c1).union(c2)
    dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
    magA = math.sqrt(sum(c1.get(k, 0)**2 for k in terms))
    magB = math.sqrt(sum(c2.get(k, 0)**2 for k in terms))
    return dotprod / (magA * magB)

def compare_lists(l1, l2):
    if len(l1) < 1 or len(l2) < 1:
        return 0.0
    l1exp = Counter()
    for itemstr in l1:
        for token in itemstr.split():
            l1exp[token] += 1
    l2exp = Counter()
    for itemstr in l2:
        for token in itemstr.split():
            l2exp[token] += 1
    return counter_cosine_similarity(l1exp, l2exp)

def compare_summaries(summaries):
    similar = []
    recorded = []
    scores = []
    for comb in combinations(summaries.keys(), 2):
        n1 = comb[0]
        n2 = comb[1]
        sim = compare_lists(summaries[n1], summaries[n2])
        if sim > 0.6:
            entry = [n1, n2, sim]
            scores.append(entry)
            found = False
            for index, entry in enumerate(similar):
                if n1 in entry or n2 in entry:
                    found = True
                    if n1 not in entry:
                        similar[index].append(n1)
                        recorded.append(n1)
                    if n2 not in entry:
                        similar[index].append(n2)
                        recorded.append(n2)
            if found == False:
                similar.append([n1, n2])
                recorded.append(n1)
                recorded.append(n2)
    not_grouped = list(Set(summaries.keys()) - Set(recorded))
    for n in not_grouped:
        similar.append([n])
    filename = os.path.join(save_dir, "similarity_scores.json")
    save_json(scores, filename)
    return similar

def cluster_tweets(raw_input_file):
    print("Preprocessing raw data")
    filename = os.path.join(save_dir, "data.json")
    processed = try_load_or_process(filename, process_raw_data, raw_input_file)
    print
    print("Unique sentences: " + str(len(processed)))

    print("Creating tag map")
    tag_map = None
    filename = os.path.join(save_dir, "tag_map.json")
    tag_map = try_load_or_process(filename, process_sentences, processed)

    clusters = cluster(tag_map, processed)
    filename = os.path.join(save_dir, "clusters.json")
    save_json(clusters, filename)

    cluster_tags = {}
    for cname, tweets in clusters.iteritems():
        cluster_tags[cname] = []
        for tweet in tweets:
            for tag in tag_map[tweet]:
                if tag not in cluster_tags[cname]:
                    cluster_tags[cname].append(tag)
    filename = os.path.join(save_dir, "cluster_tags.json")
    save_json(cluster_tags, filename)

    print("Getting summaries for each label cluster")
    summaries = summarize_cluster_map(tag_map, clusters)
    filename = os.path.join(save_dir, "cluster_summaries.json")
    save_json(summaries, filename)

    print("Looking for similar clusters.")
    similar = compare_summaries(summaries)
    filename = os.path.join(save_dir, "similar.json")
    save_json(similar, filename)



def quick_compare():
    print("Loading summaries")
    summaries = load_json(os.path.join(save_dir, "cluster_summaries.json"))
    print("Comparing summaries")
    similar = compare_summaries(summaries)
    filename = os.path.join(save_dir, "similar.json")
    save_json(similar, filename)







if __name__ == '__main__':
    base_dir = ""
    if (len(sys.argv) > 1):
        base_dir = sys.argv[1]
    else:
        print("Please provide a base directory.")
        sys.exit(0)

    g_labels = Counter()
    input_dir = os.path.join(base_dir, "data")
    save_dir = os.path.join(base_dir, "pos")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    nlp = spacy.load("en")
    cluster_min_size = 20
    cluster_word_count = 3
    vocab_count = 15
    vocab_matches = 10

    stemmer = PorterStemmer()
    all_stopwords = load_json("stopwords-iso.json")
    extra_stopwords = []
    stopwords = None
    if all_stopwords is not None:
        stopwords = all_stopwords["en"]
        stopwords += extra_stopwords

    raw_input_file = os.path.join(input_dir, "tweets.txt")
    #quick_compare()
    cluster_tweets(raw_input_file)
    test_predict()



