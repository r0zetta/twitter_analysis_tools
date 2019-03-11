# -*- coding: utf-8 -*-

from six.moves import cPickle
import os
import json
import io

def get_subdirs(d="."):
    return [os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]

def get_file_list(directory):
    ret = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.endswith('.json'):
                ret.append(f)
    return ret


def save_bin(item, filename):
    with open(filename, "wb") as f:
        cPickle.dump(item, f)

def load_bin(filename):
    ret = None
    if os.path.exists(filename):
        try:
            with open(filename, "rb") as f:
                ret = cPickle.load(f)
        except:
            pass
    return ret

def save_json(variable, filename):
    with io.open(filename, "w", encoding="utf-8") as f:
        f.write(json.dumps(variable, indent=4, ensure_ascii=False))

def load_json(filename):
    ret = None
    if os.path.exists(filename):
        try:
            with io.open(filename, "r", encoding="utf-8") as f:
                ret = json.load(f)
        except:
            pass
    return ret

def save_counter_csv(counter_data, filename):
    with io.open(filename, "w", encoding="utf-8") as f:
        for name, value in counter_data.most_common(50):
            f.write(value + "\t" + name + "\n")

def save_gephi_csv(data_map, filename):
    with io.open(filename, "w", encoding="utf-8") as f:
        f.write("Source,Target,Weight\n")
        for source, targets in data_map.items():
            if len(targets) > 0:
                for target, weight in targets.items():
                    f.write(source + "," + target + "," + str(weight) + "\n")

def save_heatmap(heatmap, filename):
    with open(filename, 'w') as handle:
        handle.write("Hour, 00, 01, 02, 03, 04, 05, 06, 07, 08, 09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23\n")
        handle.write("Mon, " + ','.join(map(str, heatmap[0])) + "\n")
        handle.write("Tue, " + ','.join(map(str, heatmap[1])) + "\n")
        handle.write("Wed, " + ','.join(map(str, heatmap[2])) + "\n")
        handle.write("Thu, " + ','.join(map(str, heatmap[3])) + "\n")
        handle.write("Fri, " + ','.join(map(str, heatmap[4])) + "\n")
        handle.write("Sat, " + ','.join(map(str, heatmap[5])) + "\n")
        handle.write("Sun, " + ','.join(map(str, heatmap[6])) + "\n")

def save_list(dataset, filename):
    with io.open(filename, "w", encoding="utf-8") as f:
        for key, val in dataset:
            f.write(val + "\t" + key + "\n")

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

def read_settings(filename):
    ret = {}
    if os.path.exists(filename):
        with open(filename, "r") as file:
            for line in file:
                if line is not None:
                    line = line.strip()
                    if len(line) > 0:
                        name, value = line.split("=")
                        name = name.strip()
                        value = int(value)
                        if value == 1:
                            ret[name] = True
                        elif value == 0:
                            ret[name] = False
    return ret

def read_config(filename, preserve_case=False):
    ret = []
    if os.path.exists(filename):
        with io.open(filename, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if preserve_case == False:
                    line = line.lower()
                if len(line) > 0:
                    ret.append(line)
    return ret

def get_stopwords(filename, lang):
    if os.path.exists(filename):
        all_stopwords = load_json(filename)
        if lang in all_stopwords:
            return all_stopwords[lang]


