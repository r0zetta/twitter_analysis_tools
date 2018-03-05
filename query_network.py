import json
import os
import sys
import io

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

def get_mentioned(names, interactions):
    ret = {}
    for name in names:
        ret[name] = []
        for source, targets in interactions.iteritems():
            if name in targets.keys():
                if source != name:
                    ret[name].append(source)
    return ret

def get_mentions(names, interactions):
        ret = {}
        for name in names:
            ret[name] = []
            if name in interactions:
                for n, v in interactions[name].iteritems():
                    ret[name].append(n)
        return ret

if __name__ == '__main__':
    if (len(sys.argv) < 3):
        print("Please supply input file path and at least one name.")
        sys.exit(0)

    input_file = sys.argv[1]
    names = sys.argv[2:]
    if not os.path.exists(input_file):
        print("Could not load " + input_file)
        sys.exit(0)
    save_dir = os.path.dirname(input_file)
    print("Save dir: " + save_dir)

    print("Loading data from " + input_file)
    interactions = load_json(input_file)
    print(str(len(names)) + " names supplied")

    print
    print("Accounts that mentioned: " + ", ".join(names))
    mentioned = get_mentioned(names, interactions)
    sets = [set(items) for key, items in mentioned.iteritems()]
    mentioned_intersection = set.intersection(*sets)
    print mentioned_intersection
    mentioned_union = set.union(*sets)

    print
    print("Accounts mentioned by: " + ", ".join(names))
    mentions = get_mentions(names, interactions)

    sets = [set(items) for key, items in mentions.iteritems()]
    mentions_intersection = set.intersection(*sets)
    print mentions_intersection
    mentions_union = set.union(*sets)

    print
    print("Intersection of above")
    all_intersection = set.intersection(mentioned_intersection, mentions_intersection)
    print all_intersection
    print
    print("Intersection of all mentioned and all mentions")
    full_intersection = set.intersection(mentioned_union, mentions_union)
    print full_intersection
    output = {}
    output["names"] = names
    output["mentioned"] = mentioned
    output["mentioned_intersection"] = list(mentioned_intersection)
    output["mentioned_union"] = list(mentioned_union)
    output["mentions"] = mentions
    output["mentions_intersection"] = list(mentions_intersection)
    output["mentions_union"] = list(mentions_union)
    output["all_intersection"] = list(all_intersection)
    output["full_intersection"] = list(full_intersection)
    filename = os.path.join(save_dir, "query_" + "_".join(sorted(names)) + ".json")
    save_json(output, filename)
