from igraph import *
from collections import Counter
import json
import os
import sys
import io

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

def make_vocab(interactions):
    names = set()
    for source, target_list in interactions.iteritems():
        names.add(source)
        for target, w in target_list.iteritems():
            names.add(target)
    vocab = {}
    for index, n in enumerate(names):
        vocab[n] = index
    return vocab

def make_graph(interactions, vocab):
    vocab_len = len(vocab)
    g = Graph()
    g.add_vertices(vocab_len)
    edge_count = 0
    max_s = len(interactions)
    count = 1
    edges = []
    print("Getting edges")
    for source, target_list in interactions.iteritems():
        print_progress(count, max_s)
        count += 1
        if len(target_list) > 0:
            for target, w in target_list.iteritems():
                edges.append((vocab[source], vocab[target]))
                edge_count += 1
    print
    print("Found " + str(vocab_len) + " nodes.")
    print("Found " + str(edge_count) + " edges.")
    print("Building graph")
    g.add_edges(edges)
    print summary(g)
    return g

def make_clusters(g, mode="multilevel"):
    print("Using " + mode + " to find communities.")
# Fast
    if "multilevel" in mode:
        return g.community_multilevel()
    if "label_propagation" in mode:
        return g.community_label_propagation()
# Slow on large graphs
    if "infomap" in mode:
        return g.community_infomap()
# Medium
    if "leading_eigenvector" in mode:
        return g.community_leading_eigenvector()
# Works only on graphs without multiple edges
    if "fastgreedy" in mode:
        return g.community_fastgreedy()
# Needs fully-connected graph
    if "spinglass" in mode:
        return g.community_spinglass()
# Returns dendrogram
    if "walktrap" in mode:
        return g.community_walktrap()
# Slow
    if "edge_betweenness" in mode:
        return g.community_edge_betweenness()
    if "optimal_modularity" in mode:
        return g.community_optimal_modularity()

def create_csv(interactions, clusters, filename):
    name_map = {}
    for cluster, names in clusters.iteritems():
        for name in names:
            name_map[name] = cluster

    with open(filename, "w") as f:
        f.write("Source,Target,Weight,Modularity Class\n")
        for source, targets in interactions.iteritems():
            if len(targets) > 1:
                for target, weight in targets.iteritems():
                    f.write(str(source) + "," + str(target) + "," + str(weight) + "," + str(name_map[target]) + "\n")





if __name__ == '__main__':
    if (len(sys.argv) != 2):
        print("No data filename supplied. Exiting.")
        sys.exit(0)
    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print("Could not load " + input_file)
        sys.exit(0)

    save_dir = os.path.dirname(input_file)

    print("Loading data from " + input_file)
    interactions = load_json(input_file)

    print("Getting vocab")
    filename = os.path.join(save_dir, "vocab.json")
    vocab = try_load_or_process(filename, make_vocab, interactions)
    vocab_inv = {}
    for name, index in vocab.iteritems():
        vocab_inv[index] = name

    g = make_graph(interactions, vocab)

    modes = ["multilevel", "label_propagation", "infomap", "leading_eigenvector"]
    for m in modes:
        clusters = make_clusters(g, m)
        print len(clusters)
        named_clusters = {}
        for index, cluster in enumerate(clusters):
            label = "%03d" % index
            named_clusters[label] = []
            for ident in cluster:
                named_clusters[label].append(vocab_inv[ident])
        filename = os.path.join(save_dir, m + "_clusters.json")
        save_json(named_clusters, filename)
        filename = os.path.join(save_dir, m + "_clusters.csv")
        create_csv(interactions, named_clusters, filename)

