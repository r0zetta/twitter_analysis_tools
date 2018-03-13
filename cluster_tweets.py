# -*- coding: utf-8 -*-
from process_text import *
from file_helpers import *

from collections import Counter
from itertools import combinations
from sets import Set
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from nltk.cluster.util import cosine_distance
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

def load_raw_data(raw_input_file):
    ret = []
    print("Loading raw data from: " + raw_input_file)
    if os.path.exists(raw_input_file):
        with io.open(raw_input_file, 'r', encoding="utf-8") as f:
            ret = f.readlines()
    return ret

def process_raw_data(input_file, lang="en"):
    ret = []
    lines = load_raw_data(input_file)
    if lines is not None and len(lines) > 0:
        num_lines = len(lines)
        for count, text in enumerate(lines):
            print_progress(count, num_lines)
            processed = preprocess_text(text, lang)
            if processed is not None:
                if processed not in ret:
                    ret.append(processed)
    return ret

def process_sentences(processed, lang, nlp, stemmer, stopwords):
    tag_map = {}
    print("Processing sentences.")
    num_sentences = len(processed)
    for count, sentence in enumerate(processed):
        print_progress(count + 1, num_sentences)
        tags = process_sentence(sentence, lang, nlp, stemmer, stopwords)
        if tags is not None:
            tag_map[sentence] = tags
        else:
            tag_map[sentence] = []
    return tag_map

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

def cluster(tag_map, tweets, lang="en"):
    print("Calculating frequency distribution")
    filename = os.path.join(save_dir, "freq_dist.json")
    if os.path.exists(filename):
        dist = Counter(load_json(filename))
    else:
        dist = get_freq_dist(tag_map, lang)
        save_json(dist, filename)
    filename = os.path.join(save_dir, "freq_dist_common.json")
    with io.open(filename, "w", encoding="utf-8") as f:
        for word, count in dist.most_common(200):
            f.write(unicode(count) + "\t" + word + u"\n")

    min_hits = 2
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
    num_k = 60
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
    all_sims = []
    average_sim = 0
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 != idx2:
                sim = sentence_similarity(sentences[idx1], sentences[idx2])
                S[idx1][idx2] = sim
                all_sims.append(sim)
 
    for idx in range(len(S)):
        if S[idx].sum() != 0:
            S[idx] /= S[idx].sum()
    if sum(all_sims) > 0 and len(all_sims) > 0:
        average_sim = sum(all_sims) / float(len(all_sims))
    return S, average_sim

def textrank(tag_map, sentences, top_n=5):
    if len(sentences) < 1:
        return [], 0
    tokenized = []
    for s in sentences:
        tokenized.append(tag_map[s])
    S, average_sim = build_similarity_matrix(tokenized)
    if not np.all(S):
        sentence_ranks = pagerank(S)
        ranked_sentence_indexes = [item[0] for item in sorted(enumerate(sentence_ranks), key=lambda item: -item[1])]
        selected_sentences = sorted(ranked_sentence_indexes[:top_n])
        summary = []
        for index in selected_sentences:
            summary.append(sentences[index])
        return summary, average_sim
    else:
        return [], average_sim

def summarize_cluster_map(tag_map, cluster_map):
    summaries = {}
    average_sim = {}
    filename1 = os.path.join(save_dir, "temp_summaries.json")
    filename2 = os.path.join(save_dir, "temp_average_sim.json")
    max_s = len(cluster_map)
    count = 0
    for cluster, tweets in cluster_map.iteritems():
        count += 1
        print_progress(count, max_s)
        summ, av =  textrank(tag_map, tweets)
        summaries[cluster] = summ
        average_sim[cluster] = av
        save_json(summaries, filename1)
        save_json(average_sim, filename2)
    return summaries, average_sim

def test_predict(lang="en"):
    filename = os.path.join(config_dir, "stopwords.json")
    stopwords = get_stopwords(filename, lang)
    nlp, stemmer = init_nlp_single_lang(lang)

    filename = os.path.join(save_dir, "common_vocab.json")
    common_vocab = load_json(filename)

    pca = PCA()
    filename = os.path.join(save_dir, "pca_model.sav")
    pca = pickle.load(open(filename, "rb"))

    model = KMeans()
    filename = os.path.join(save_dir, "k_means_model.sav")
    model = pickle.load(open(filename, "rb"))

    filename = os.path.join(save_dir, "data.json")
    processed = []
    if os.path.exists(filename):
        processed = load_json(filename)
    else:
        processed = process_raw_data(raw_input_file, lang)
        save_json(processed, filename)
    print("Testing predict function.")
    for n in range(20):
        tweet = processed[randint(0,len(processed)-1)]
        print tweet
        category = predict_tweet(tweet, lang, nlp, stemmer, stopwords, common_vocab, pca, model)
        print category

def predict_tweet(tweet, lang, nlp, stemmer, stopwords, common_vocab, pca, model):
    tweet = preprocess_text(tweet)
    if tweet is not None:
        tags = process_sentence(tweet, lang, nlp, stemmer, stopwords)
        if tags is not None:
            row = vectorize_item(tags, common_vocab)

            Y = np.array(row)
            Y = pca.transform(Y.reshape(1,-1))

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

def cluster_tweets(raw_input_file, lang, nlp, stemmer, stopwords):
    print("Preprocessing raw data")
    processed = []
    filename = os.path.join(save_dir, "data.json")
    if os.path.exists(filename):
        processed = load_json(filename)
    else:
        processed = process_raw_data(raw_input_file, lang)
        save_json(processed, filename)
    print
    print("Unique sentences: " + str(len(processed)))

    print("Creating tag map")
    tag_map = None
    filename = os.path.join(save_dir, "tag_map.json")
    if os.path.exists(filename):
        tag_map = load_json(filename)
    else:
        tag_map = process_sentences(processed, lang, nlp, stemmer, stopwords)
        save_json(tag_map, filename)

    print("Creating clusters")
    clusters = None
    filename = os.path.join(save_dir, "clusters.json")
    if os.path.exists(filename):
        clusters = load_json(filename)
    else:
        clusters = cluster(tag_map, processed, lang)
        save_json(clusters, filename)

    print("Creating cluster tag map")
    cluster_tags = {}
    filename = os.path.join(save_dir, "cluster_tags.json")
    if os.path.exists(filename):
        cluster_tags = load_json(filename)
    else:
        for cname, tweets in clusters.iteritems():
            cluster_tags[cname] = []
            for tweet in tweets:
                for tag in tag_map[tweet]:
                    if tag not in cluster_tags[cname]:
                        cluster_tags[cname].append(tag)
        filename = os.path.join(save_dir, "cluster_tags.json")
        save_json(cluster_tags, filename)

    print("Getting summaries for each cluster")
    summaries = {}
    average_sim = {}
    filename1 = os.path.join(save_dir, "cluster_summaries.json")
    filename2 = os.path.join(save_dir, "cluster_average_sim.json")
    if os.path.exists(filename1) and os.path.exists(filename2):
        summaries = load_json(filename1)
        average_sim = load_json(filename2)
    else:
        summaries, average_sim = summarize_cluster_map(tag_map, clusters)
        save_json(summaries, filename1)
        save_json(average_sim, filename2)

    print("Looking for similar clusters.")
    similar = compare_summaries(summaries)
    filename = os.path.join(save_dir, "similar.json")
    save_json(similar, filename)

    print("Consolidating similar clusters")
    map_to_consolidated = {}
    similar_summaries = {}
    similar_average_sim = {}
    for index, similar_clusters in enumerate(similar):
        new_label = "%02d" % index
        map_to_consolidated[new_label] = []
        if len(similar_clusters) > 1:
            cluster_summaries = []
            cluster_average_sims = []
            for label in similar_clusters:
                map_to_consolidated[new_label].append(label)
                for s in summaries[label]:
                    cluster_summaries.append(s)
                cluster_average_sims.append(average_sim[label])
            new_summaries, new_average_sim = textrank(tag_map, cluster_summaries)
            if sum(cluster_average_sims) > 0 and len(cluster_average_sims) > 0:
                new_average_sim = sum(cluster_average_sims) / float(len(cluster_average_sims))
            similar_summaries[new_label] = new_summaries
            similar_average_sim[new_label] = new_average_sim
        elif len(similar_clusters) > 0:
            map_to_consolidated[new_label].append(similar_clusters[0])
            similar_summaries[new_label] = summaries[similar_clusters[0]]
            similar_average_sim[new_label] = average_sim[similar_clusters[0]]
    filename = os.path.join(save_dir, "final_cluster_summaries.json")
    save_json(similar_summaries, filename)
    filename = os.path.join(save_dir, "final_cluster_average_sim.json")
    save_json(similar_average_sim, filename)




def quick_compare():
    print("Loading summaries")
    summaries = load_json(os.path.join(save_dir, "cluster_summaries.json"))
    print("Comparing summaries")
    similar = compare_summaries(summaries)
    filename = os.path.join(save_dir, "similar.json")
    save_json(similar, filename)







if __name__ == '__main__':
    cluster_min_size = 20
    cluster_word_count = 3
    vocab_count = 15
    vocab_matches = 10

    config_dir = "config"
    input_dir = "data/raw"
    save_dir = "clusters"
    base_dir = None
    if (len(sys.argv) > 1):
        base_dir = sys.argv[1]

    if base_dir is not None:
        input_dir = os.path.join(base_dir, "data")
        save_dir = os.path.join(base_dir, "clusters")
        config_dir = ""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    lang = "en"
    filename = os.path.join(config_dir, "languages.txt")
    if os.path.exists(filename):
        langs = read_config(filename)
        if len(langs) > 0:
            lang = langs[0]
    print("Got lang: " + lang)

    filename = os.path.join(config_dir, "stopwords.json")
    stopwords = get_stopwords(filename, lang)
    print("Stopwords length: " + str(len(stopwords)))
    nlp, stemmer = init_nlp_single_lang(lang)

    raw_input_file = os.path.join(input_dir, "tweets.txt")
    #quick_compare()
    cluster_tweets(raw_input_file, lang, nlp, stemmer, stopwords)
    test_predict(lang)



