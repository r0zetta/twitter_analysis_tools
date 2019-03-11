from igraph import *
import sys, os, io, random, json, re
import pandas as pd
from collections import Counter
from time_helpers import *

report_every = 100000

#####################################
# Basic helper functions
#####################################
def find_exact_string(w):
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

def load_json(fn):
    ret = None
    with io.open(fn, "r", encoding="utf-8") as f:
        ret = json.load(f)
    return ret

def load_jsons(filename):
    ret = []
    with io.open(filename, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            ret.append(entry)
    return ret

def save_json(d, fn):
    with io.open(fn, "w", encoding="utf-8") as f:
        f.write(json.dumps(d, indent=4))

def save_csv(inter, fn):
    with io.open(fn, "w", encoding="utf-8") as f:
        f.write("Source,Target,Weight\n")
        for source, targets in inter.items():
            for target, count in targets.items():
                f.write(source + "," + target + "," + str(count) + "\n")

def make_batches(data, batch_len):
    num_batches = int(len(data)/batch_len)
    batches = (data[i:i+batch_len] for i in range(0, len(data), batch_len))
    return batches

def print_tweet_list(tweet_list, tweet_url_map):
    for t in tweet_list:
        print(t + "\t" + tweet_url_map[t])

def print_sn_tweet_list(sn_tweet_list, tweet_url_map):
    for sn, tweets in sn_tweet_list.items():
        print("")
        print("https://twitter.com/" + sn)
        print("================================")
        for t in tweets:
            print("\t" + t)
            print("\t" + tweet_url_map[t])
            print("")

def print_tweet_counter(tweet_counter, tweet_url_map, num):
    for tweet, count in tweet_counter.most_common(num):
        print(str(count) + "\t" + tweet + "\t" + tweet_url_map[tweet])

def print_tweet_texts(twid_count, twid_text, twid_url, num):
    for twid, count in twid_count.most_common(num):
        text = twid_text[twid]
        url = twid_url[twid]
        print(str(count) + "\t" + text + "\t" + url)

def print_sn_list(sn_list):
    for sn in sn_list:
        print("https://twitter.com/" + sn)

def print_sn_counter(sn_list, num):
    for user, count in sn_list.most_common(num):
        snstr = "https://twitter.com/" + user
        print(str(count) + "\t" + snstr)

def print_hashtag_list(ht_list):
    for ht in ht_list:
        print("https://twitter.com/search?q=%23" + ht)

def print_hashtag_counter(ht_list, num):
    for ht, count in ht_list.most_common(num):
        htstr = "https://twitter.com/search?q=%23" + ht
        print(str(count) + "\t" + htstr)

def print_counter(ct, num):
    for item, count in ct.most_common(num):
        print(str(count) + "\t" + item)

def print_counters(counters, user_fields, list_len):
    counter_names = [x for x, y in counters.items()]
    for n in counter_names:
        print("")
        print(n)
        print("---------")
        if n in user_fields:
            print_sn_counter(counters[n], list_len)
        elif n == "hashtags":
            print_hashtag_counter(counters[n], list_len)
        else:
            print_counter(counters[n], list_len)

def tokenize_sentence(text, stopwords):
    text = text.replace(",", "")
    text = text.replace(".", "")
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
        return []
# Remove stopwords and other undesirable tokens
    cleaned = []
    endtoks = [":", ",", "\'", "…", "."]
    for token in tokens:
        if len(token) > 0:
            if stopwords is not None:
                if token in stopwords:
                    token = None
            if token is not None and len(token) > 0 and token[-1] in endtoks:
                token = token[0:-1]
            if token is not None and len(token) > 0 and token[0] in endtoks:
                token = token[1:]
            if token is not None:
                if re.search(".+…$", token):
                    token = None
            if token is not None:
                if token == "#":
                    token = None
            if token is not None:
                cleaned.append(token)
    if len(cleaned) < 1:
        return []
    return cleaned

#####################################
# Data loading and caching
#####################################
def read_from_raw_data(start_time, end_time, filename):
    ret = []
    print("Reading data from: " + start_time + " to: " + end_time)
    start = time_object_to_unix(time_string_to_object(start_time))
    end = time_object_to_unix(time_string_to_object(end_time))
    count = 0
    with io.open(filename, "r", encoding="utf-8") as f:
        for line in f:
            count += 1
            if count % report_every == 0:
                print("Count: " + str(count))
            d = json.loads(line)
            timestamp = twitter_time_to_unix(d["created_at"])
            if timestamp >= end:
                break
            if timestamp >= start and timestamp <= end:
                ret.append(d)
    print("Saw a total of " + str(count) + " records.")
    print(str(len(ret)) + " records matched the date range.")
    return ret

def create_cache(start_time, end_time, filename, cachefile):
    print("Creating cache with data from: " + start_time + " to: " + end_time)
    start = time_object_to_unix(time_string_to_object(start_time))
    end = time_object_to_unix(time_string_to_object(end_time))
    cf = io.open(cachefile, "a", encoding="utf-8")
    count = 0
    matched = 0
    with io.open(filename, "r", encoding="utf-8") as f:
        for line in f:
            count += 1
            if count % report_every == 0:
                print("Count: " + str(count))
            d = json.loads(line)
            timestamp = twitter_time_to_unix(d["created_at"])
            if timestamp >= end:
                break
            if timestamp >= start and timestamp <= end:
                matched += 1
                cf.write(line)
    cf.close()
    print("Cached a total of " + str(matched) + " records.")
    return cachefile

def make_file_iterator(start_time, end_time, filename="data/raw.json"):
    print("Creating iterator from: " + start_time + " to: " + end_time)
    start = time_object_to_unix(time_string_to_object(start_time))
    end = time_object_to_unix(time_string_to_object(end_time))
    count = 0
    with io.open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if len(line) == 0:
                print("Resetting iterator")
                f.seek(0)
                return
            count += 1
            if count % report_every == 0:
                print("Count: " + str(count))
            d = json.loads(line)
            timestamp = twitter_time_to_unix(d["created_at"])
            if timestamp >= end:
                break
            if timestamp >= start and timestamp <= end:
                yield(d)

def read_timeline_data(filename):
    if not os.path.exists(filename):
        print("File: " + filename + " did not exist.")
        return
    print("Reading in " + filename)
    count = 0
    ret = []
    with io.open(filename, "r", encoding="utf-8") as f:
        for line in f:
            count += 1
            if count % 1000 == 0:
                print("Count: " + str(count))
            d = json.loads(line)
            ret.append(d)
    return ret

def make_timeline_iterator(filename):
    if not os.path.exists(filename):
        print("File: " + filename + " did not exist.")
        return
    print("Opening iterator on " + filename)
    count = 0
    with io.open(filename, "r", encoding="utf-8") as f:
        for line in f:
            count += 1
            if count % 1000 == 0:
                print("Count: " + str(count))
            d = json.loads(line)
            yield(d)

def get_timelines(filename):
    all_tweets = []
    seen = []
    raw_data = make_timeline_iterator(filename)
    for item in raw_data:
        sn = list(item.keys())[0]
        if sn.lower() not in seen:
            seen.append(sn.lower())
            tweets = list(item.values())[0]
            tweet_count = 0
            for d in tweets:
                all_tweets.append(d)
                tweet_count += 1
        print("https://twitter.com/" + sn + " : " + str(tweet_count))
    return all_tweets
#####################################
# Graph manipulation
#####################################
def match_graph(inter, sn_list):
    ret = {}
    for source, targets in inter.items():
        if source in sn_list:
            ret[source] = targets
            continue
        for target, weight in targets.items():
            if target in sn_list:
                ret[source] = targets
                break
    return ret

def match_graph2(inter, sn_list):
    ret = {}
    for source, targets in inter.items():
        if source in sn_list:
            for target, weight in targets.items():
                if target in sn_list:
                    if source not in ret:
                        ret[source] = {}
                    ret[source][target] = weight
    return ret

def trim_graph(inter, all_users, cutoff):
    print("All users: " + str(len(all_users)))
    trimmed_users = []
    for name, count in all_users.items():
        if count > cutoff:
            trimmed_users.append(name)
    print("Trimmed users: " + str(len(trimmed_users)))
    trimmed = {}
    for source, targets in inter.items():
        if source in trimmed_users:
            trimmed[source] = targets
            continue
        found = False
        for target, count in targets.items():
            if target in trimmed_users:
                trimmed[source] = targets
                break
    return trimmed

def trim_graph_influencers(inter, influencers, cutoff):
    print("Influencers: " + str(len(influencers)))
    trimmed_users = []
    for name, count in influencers.items():
        if count > cutoff:
            trimmed_users.append(name)
    print("Trimmed users: " + str(len(trimmed_users)))
    trimmed = {}
    for source, targets in inter.items():
        if source in trimmed_users:
            trimmed[source] = targets
            continue
        found = False
        for target, count in targets.items():
            if target in trimmed_users:
                trimmed[source] = targets
                break
    return trimmed

def trim_graph2(inter, all_users, cutoff):
    print("All users: " + str(len(all_users)))
    trimmed_users = []
    for name, count in all_users.items():
        if count > cutoff:
            trimmed_users.append(name)
    print("Trimmed users: " + str(len(trimmed_users)))
    trimmed = {}
    for source, targets in inter.items():
        if source in trimmed_users:
            for target, count in targets.items():
                if target in trimmed_users:
                    if source not in trimmed:
                        trimmed[source] = {}
                    trimmed[source][target] = count
            continue
        found = False
        for target, count in targets.items():
            if target in trimmed_users:
                if source not in trimmed:
                    trimmed[source] = {}
                trimmed[source][target] = count
    return trimmed

def get_communities(inter):
    names = set()
    print("Building vocab")
    for source, targets in inter.items():
        names.add(source)
        for target, count in targets.items():
            names.add(target)
    vocab = {}
    vocab_inv = {}
    for index, name in enumerate(names):
        vocab[name] = index
        vocab_inv[index] = name
    print("Vocab length: " + str(len(vocab)))

    vocab_len = len(vocab)
    g = Graph()
    g.add_vertices(vocab_len)
    edge_count = 0
    max_s = len(inter)
    edges = []
    print("Getting edges")
    for source, target_list in inter.items():
        if len(target_list) > 0:
            for target, w in target_list.items():
                edges.append((vocab[source], vocab[target]))
                edge_count += 1
    print
    print("Found " + str(vocab_len) + " nodes.")
    print("Found " + str(edge_count) + " edges.")
    print("Building graph")
    g.add_edges(edges)
    print(summary(g))

    print("Getting communities.")
    communities = g.community_multilevel()
    print("Found " + str(len(communities)) + " communities.")

    clusters = {}
    for mod, nodelist in enumerate(communities):
        clusters[mod] = []
        for ident in nodelist:
            clusters[mod].append(vocab_inv[ident])
    return clusters

def get_cluster_for_sn(sn, clusters):
    for index, names in clusters.items():
        if sn in names:
            return names

#####################################
# Searches that return plottable data
#####################################
def plot_user_activity(raw_data, userlist):
    timestamps = Counter()
    for d in raw_data:
        sn = d["user"]["screen_name"]
        if sn in userlist:
            tobj = twitter_time_to_readable(d["created_at"])
            thour = tobj[:-3]
            timestamps[thour] += 1
    df = pd.Series(timestamps)
    return df

def plot_retweet_activity(raw_data, sn_list):
    timestamps = {}
    for sn in sn_list:
        timestamps[sn] = {}
    for d in raw_data:
        if "retweeted_status" in d:
            r = d["retweeted_status"]
            sn = r["user"]["screen_name"]
            if sn in sn_list:
                tobj = twitter_time_to_readable(d["created_at"])
                thour = tobj[:-3]
                if thour not in timestamps[sn]:
                    timestamps[sn][thour] = 1
                else:
                    timestamps[sn][thour] += 1
    df = None
    if len(sn_list) > 1:
        df = pd.DataFrame(timestamps)
    else:
        df = pd.Series(timestamps)
    df = df.interpolate()
    return df

def plot_url_trends(raw_data, urls):
    timestamps = {}
    for u in urls:
        timestamps[u] = {}
    for d in raw_data:
        if "urls" in d and d["urls"] is not None and len(d["urls"]) > 0:
            for u in urls:
                if u in d["urls"]:
                    tobj = twitter_time_to_readable(d["created_at"])
                    thour = tobj[:-3]
                    if thour not in timestamps[u]:
                        timestamps[u][thour] = 1
                    else:
                        timestamps[u][thour] += 1
    df = None
    if len(urls) > 1:
        df = pd.DataFrame(timestamps)
    else:
        df = pd.Series(timestamps)
    df = df.interpolate()
    return df

def plot_user_trends(raw_data, users):
    timestamps = {}
    for u in users:
        timestamps[u] = {}
    for d in raw_data:
        if "user" in d and "screen_name" in d["user"]:
            sn = d["user"]["screen_name"]
            for u in users:
                if u == sn:
                    tobj = twitter_time_to_readable(d["created_at"])
                    thour = tobj[:-3]
                    if thour not in timestamps[u]:
                        timestamps[u][thour] = 1
                    else:
                        timestamps[u][thour] += 1
    df = None
    if len(users) > 1:
        df = pd.DataFrame(timestamps)
    else:
        df = pd.Series(timestamps)
    df = df.interpolate()
    return df

def plot_hashtag_trends(raw_data, hashtags):
    timestamps = {}
    for h in hashtags:
        timestamps[h] = {}
    count = 0
    for d in raw_data:
        if "hashtags" in d:
            ht = d["hashtags"]
            for h in hashtags:
                if h in ht:
                    tobj = twitter_time_to_readable(d["created_at"])
                    thour = tobj[:-3]
                    if thour not in timestamps[h]:
                        timestamps[h][thour] = 1
                    else:
                        timestamps[h][thour] += 1
    df = None
    if len(hashtags) > 1:
        df = pd.DataFrame(timestamps)
    else:
        df = pd.Series(timestamps)
    df = df.interpolate()
    return df

#####################################
# Generic analysis of data
#####################################
def make_hashtag_interactions(raw_data, ht_list, whitelist):
    interactions = {}
    for d in raw_data:
        sn = d["user"]["screen_name"]
        if sn in whitelist:
            continue
        sn = "@" + sn
        if "hashtags" in d:
            hashtags = d["hashtags"]
            if len(set(hashtags).intersection(set(ht_list))) > 0:
                if sn not in interactions:
                    interactions[sn] = {}
                if "retweeted_status" in d:
                    rsn = d["retweeted_status"]["user"]["screen_name"]
                    if rsn not in whitelist:
                        rsn = "@" + rsn
                        if rsn not in interactions[sn]:
                            interactions[sn][rsn] = 1
                        else:
                            interactions[sn][rsn] += 1
                for h in hashtags:
                    ht = "#" + h
                    if ht not in interactions[sn]:
                        interactions[sn][ht] = 1
                    else:
                        interactions[sn][ht] += 1
    return interactions

def get_counters_and_interactions_whitelist(raw_data, whitelist):
    interactions = {}
    counters = {}

    stopwords = load_json("config/stopwords.json")
    stopwords = stopwords["en"]
    stopwords += ["rt", "-", "&amp;", "|"]

    counter_names = ["users",
                     "influencers",
                     "amplifiers",
                     "hashtags",
                     "highly_retweeted_users",
                     "highly_replied_to_users",
                     "words",
                     "urls"]
    user_fields = ["users",
                   "influencers",
                   "amplifiers",
                   "highly_retweeted_users",
                   "highly_replied_to_users"]
    for n in counter_names:
        counters[n] = Counter()
    count = 0
    for d in raw_data:
        count += 1
        twid = d["id_str"]
        sn = d["user"]["screen_name"]
        if sn in whitelist:
            continue
        text = d["text"]
        tokens = tokenize_sentence(text, stopwords)
        for t in tokens:
            counters["words"][t] += 1
        if "retweeted_status" in d:
            rsn = d["retweeted_status"]["user"]["screen_name"]
            if rsn in whitelist:
                continue
            counters["highly_retweeted_users"][rsn] += 1
            counters["amplifiers"][sn] += 1
            counters["influencers"][rsn] += 1
            if sn not in interactions:
                interactions[sn] = {}
            if rsn not in interactions[sn]:
                interactions[sn][rsn] = 1
            else:
                interactions[sn][rsn] += 1
        counters["users"][sn] += 1
        if "in_reply_to_screen_name" in d and d["in_reply_to_screen_name"] is not None:
            counters["highly_replied_to_users"][d["in_reply_to_screen_name"].lower()] += 1
        if "hashtags" in d:
            ht = [x.lower() for x in d["hashtags"]]
            for h in ht:
                counters["hashtags"][h] += 1
        if "urls" in d:
            urls = d["urls"]
            for u in urls:
                if "twitter" not in u:
                    counters["urls"][u] += 1
    print("Processed " + str(count) + " tweets.")
    print("Found " + str(len(counters["users"])) + " users.")
    print("Found " + str(len(counters["hashtags"])) + " hashtags.")
    print("Found " + str(len(counters["urls"])) + " urls.")
    print("Found " + str(len(counters["amplifiers"])) + " amplifiers.")
    print("Found " + str(len(counters["influencers"])) + " influencers.")
    return user_fields, counters, interactions

def get_counters_and_interactions_retweet_only(raw_data):
    interactions = {}
    counters = {}

    stopwords = load_json("config/stopwords.json")
    stopwords = stopwords["en"]
    stopwords += ["rt", "-", "&amp;", "|"]

    counter_names = ["users",
                     "influencers",
                     "amplifiers",
                     "hashtags",
                     "highly_retweeted_users",
                     "highly_replied_to_users",
                     "words",
                     "urls"]
    user_fields = ["users",
                   "influencers",
                   "amplifiers",
                   "highly_retweeted_users",
                   "highly_replied_to_users"]
    for n in counter_names:
        counters[n] = Counter()
    count = 0
    for d in raw_data:
        count += 1
        twid = d["id_str"]
        sn = d["user"]["screen_name"]
        counters["users"][sn] += 1
        text = d["text"]
        tokens = tokenize_sentence(text, stopwords)
        for t in tokens:
            counters["words"][t] += 1
        if "retweeted_status" in d:
            rsn = d["retweeted_status"]["user"]["screen_name"]
            counters["highly_retweeted_users"][rsn] += 1
            counters["amplifiers"][sn] += 1
            counters["influencers"][rsn] += 1
            if sn not in interactions:
                interactions[sn] = {}
            if rsn not in interactions[sn]:
                interactions[sn][rsn] = 1
            else:
                interactions[sn][rsn] += 1
        if "in_reply_to_screen_name" in d and d["in_reply_to_screen_name"] is not None:
            counters["highly_replied_to_users"][d["in_reply_to_screen_name"].lower()] += 1
        if "hashtags" in d:
            ht = [x.lower() for x in d["hashtags"]]
            for h in ht:
                counters["hashtags"][h] += 1
        if "urls" in d:
            urls = d["urls"]
            for u in urls:
                if "twitter" not in u:
                    counters["urls"][u] += 1
    print("Processed " + str(count) + " tweets.")
    print("Found " + str(len(counters["users"])) + " users.")
    print("Found " + str(len(counters["hashtags"])) + " hashtags.")
    print("Found " + str(len(counters["urls"])) + " urls.")
    print("Found " + str(len(counters["amplifiers"])) + " amplifiers.")
    print("Found " + str(len(counters["influencers"])) + " influencers.")
    return user_fields, counters, interactions

def get_counters_and_interactions(raw_data):
    interactions = {}
    counters = {}

    stopwords = load_json("config/stopwords.json")
    stopwords = stopwords["en"]
    stopwords += ["rt", "-", "&amp;"]

    counter_names = ["users",
                     "influencers",
                     "amplifiers",
                     "hashtags",
                     "highly_retweeted_users",
                     "highly_replied_to_users",
                     "repliers",
                     "words",
                     "urls"]
    user_fields = ["users",
                   "influencers",
                   "amplifiers",
                   "repliers",
                   "highly_retweeted_users",
                   "highly_replied_to_users"]
    for n in counter_names:
        counters[n] = Counter()
    count = 0
    for d in raw_data:
        count += 1
        twid = d["id_str"]
        sn = d["user"]["screen_name"].lower()
        counters["users"][sn] += 1
        text = d["text"]
        tokens = tokenize_sentence(text, stopwords)
        for t in tokens:
            counters["words"][t] += 1
        if "retweeted_status" in d:
            retweeted_sn = d["retweeted_status"]["user"]["screen_name"].lower()
            counters["highly_retweeted_users"][retweeted_sn] += 1
        if "in_reply_to_screen_name" in d and d["in_reply_to_screen_name"] is not None:
            counters["highly_replied_to_users"][d["in_reply_to_screen_name"].lower()] += 1
            counters["repliers"][sn] += 1
        if "hashtags" in d:
            ht = [x.lower() for x in d["hashtags"]]
            for h in ht:
                counters["hashtags"][h] += 1
        if "urls" in d:
            urls = d["urls"]
            for u in urls:
                if "twitter" not in u:
                    counters["urls"][u] += 1
        if "interactions" in d:
            counters["amplifiers"][sn] += 1
            inter = [x.lower() for x in d["interactions"]]
            if sn not in interactions:
                interactions[sn] = {}
            for i in inter:
                counters["influencers"][i] += 1
                if i not in interactions[sn]:
                    interactions[sn][i] = 1
                else:
                    interactions[sn][i] += 1
    print("Processed " + str(count) + " tweets.")
    print("Found " + str(len(counters["users"])) + " users.")
    print("Found " + str(len(counters["hashtags"])) + " hashtags.")
    print("Found " + str(len(counters["urls"])) + " urls.")
    print("Found " + str(len(counters["amplifiers"])) + " amplifiers.")
    print("Found " + str(len(counters["influencers"])) + " influencers.")
    print("Found " + str(len(counters["repliers"])) + " repliers.")
    return user_fields, counters, interactions

def get_highly_interacted(raw_data, cutoff):
    highly_retweeted = []
    highly_retweeted_ids = []
    highly_liked = []
    highly_liked_ids = []
    highly_replied = []
    highly_replied_ids = []
    highly_retweeted_users = Counter()
    highly_replied_to_users = Counter()

    for d in raw_data:
        if "reply_count" in d:
            if d["reply_count"] is not None and d["reply_count"] > cutoff:
                if twid not in highly_replied_ids:
                    highly_replied_ids.append(twid)
                    highly_replied.append(d)
        if "retweet_count" in d:
            if d["retweet_count"] is not None and d["retweet_count"] > cutoff:
                if twid not in highly_retweeted_ids:
                    highly_retweeted_ids.append(twid)
                    highly_retweeted.append(d)
        if "favorite_count" in d:
            if d["favorite_count"] is not None and d["favorite_count"] > cutoff:
                if twid not in highly_liked_ids:
                    highly_liked_ids.append(twid)
                    highly_liked.append(d)
        if "retweeted_status" in d:
            rtwid = d["retweeted_status"]["id_str"]
            retweeted_sn = d["retweeted_status"]["user"]["screen_name"]
            s = d["retweeted_status"]
            if "retweet_count" in s:
                if s["retweet_count"] is not None and s["retweet_count"] > cutoff:
                    if rtwid not in highly_retweeted_ids:
                        highly_retweeted_ids.append(rtwid)
                        highly_retweeted.append(s)
            if "favorite_count" in s:
                if s["favorite_count"] is not None and s["favorite_count"] > cutoff:
                    if rtwid not in highly_liked_ids:
                        highly_liked_ids.append(rtwid)
                        highly_liked.append(s)
            if "reply_count" in s:
                if s["reply_count"] is not None and s["reply_count"] > cutoff:
                    if twid not in highly_replied_ids:
                        highly_replied_ids.append(twid)
                        highly_replied.append(s)
    print("Highly retweeted: " + str(len(highly_retweeted)))
    print("Highly liked: " + str(len(highly_liked)))
    print("Highly replied to: " + str(len(highly_replied)))
    return highly_retweeted, highly_liked, highly_replied

def get_tweet_id_interactions(raw_data, twid_list):
    interactions = {}
    for d in raw_data:
        sn = d["user"]["screen_name"]
        twid = d["id_str"]
        if "retweeted_status" in d:
            twid = d["retweeted_status"]["id_str"]
        if twid in twid_list:
            if sn not in interactions:
                interactions[sn] = {}
            interactions[sn][twid] = 1
    return interactions

def get_counters_and_interactions2(raw_data):
    twid_count = Counter()
    twid_rtc = Counter()
    twid_text = {}
    twid_url = {}
    twid_sn = {}
    sn_twid = {}
    sn_hashtag = {}
    hashtag_sn = {}
    sn_url = {}
    url_sn = {}

    sn_rsn = {}
    rsn_sn = {}
    sn_rep = {}
    rep_sn = {}
    sn_men = {}
    men_sn = {}
    counters = {}

    stopwords = load_json("config/stopwords.json")
    stopwords = stopwords["en"]
    stopwords += ["rt", "-", "&amp;"]

    counter_names = ["users",
                     "influencers",
                     "amplifiers",
                     "hashtags",
                     "mentioned",
                     "mentioners",
                     "retweeted",
                     "retweeters",
                     "replied_to",
                     "repliers",
                     "words",
                     "urls"]
    user_fields = ["users",
                   "influencers",
                   "amplifiers",
                   "mentioned",
                   "mentioners",
                   "repliers",
                   "retweeted",
                   "retweeters",
                   "replied_to"]
    for n in counter_names:
        counters[n] = Counter()
    count = 0
    for d in raw_data:
        count += 1
        twid = d["id_str"]
        sn = d["user"]["screen_name"]
        counters["users"][sn] += 1
        text = d["text"]
        twid = d["id_str"]
        rtc = d["retweet_count"]
        url = "https://twitter.com/" + sn + "/status/" + twid
        tokens = tokenize_sentence(text, stopwords)
        for t in tokens:
            counters["words"][t] += 1
        if "retweeted_status" in d:
            text = d["retweeted_status"]["text"]
            twid = d["retweeted_status"]["id_str"]
            rtc = d["retweeted_status"]["retweet_count"]
            rsn = d["retweeted_status"]["user"]["screen_name"]
            url = "https://twitter.com/" + rsn + "/status/" + twid
            counters["retweeted"][rsn] += 1
            counters["retweeters"][sn] += 1
            counters["amplifiers"][sn] += 1
            counters["influencers"][rsn] += 1
            if sn not in sn_rsn:
                sn_rsn[sn] = Counter()
            sn_rsn[sn][rsn] += 1
            if rsn not in rsn_sn:
                rsn_sn[rsn] = Counter()
            rsn_sn[rsn][sn] += 1
        twid_count[twid] += 1
        twid_rtc[twid] = rtc
        twid_text[twid] = text.replace("\n", "")
        twid_url[twid] = url
        if "mentioned" in d and d["mentioned"] is not None:
            if len(d["mentioned"]) > 0:
                if sn not in sn_men:
                    sn_men[sn] = Counter()
                for m in d["mentioned"]:
                    if m not in men_sn:
                        men_sn[m] = Counter()
                    counters["mentioned"][m] += 1
                    counters["mentioners"][sn] += 1
                    sn_men[sn][m] += 1
                    men_sn[m][sn] += 1
                    counters["amplifiers"][sn] += 1
                    counters["influencers"][m] += 1
        if twid not in twid_sn:
            twid_sn[twid] = Counter()
        twid_sn[twid][sn] += 1
        if sn not in sn_twid:
            sn_twid[sn] = Counter()
        sn_twid[sn][twid] += 1
        if "in_reply_to_screen_name" in d and d["in_reply_to_screen_name"] is not None:
            rep = d["in_reply_to_screen_name"]
            if sn not in sn_rep:
                sn_rep[sn] = Counter()
            sn_rep[sn][rep] += 1
            if rep not in rep_sn:
                rep_sn[rep] = Counter()
            rep_sn[rep][sn] += 1
            counters["replied_to"][rep] += 1
            counters["repliers"][sn] += 1
            counters["amplifiers"][sn] += 1
            counters["influencers"][rep] += 1
        if "hashtags" in d:
            ht = d["hashtags"]
            if len(ht) > 0:
                if sn not in sn_hashtag:
                    sn_hashtag[sn] = Counter()
                for h in ht:
                    if h not in hashtag_sn:
                        hashtag_sn[h] = Counter()
                    hashtag_sn[h][sn] += 1
                    sn_hashtag[sn][h] += 1
                    counters["hashtags"][h] += 1
        if "urls" in d:
            urls = d["urls"]
            if len(urls) > 0:
                if sn not in sn_url:
                    sn_url[sn] = Counter()
                for u in urls:
                    if u not in url_sn:
                        url_sn[u] = Counter()
                    url_sn[u][sn] += 1
                    sn_url[sn][u] += 1
                    if "twitter" not in u:
                        counters["urls"][u] += 1
    interactions = {}
    interactions ["sn_rsn"] = sn_rsn
    interactions ["sn_rep"] = sn_rep
    interactions ["sn_men"] = sn_men
    interactions ["rep_sn"] = rep_sn
    interactions ["rsn_sn"] = rsn_sn
    interactions ["men_sn"] = men_sn
    twid_assoc = {}
    twid_assoc["twid_count"] = twid_count
    twid_assoc["twid_rtc"] = twid_rtc
    twid_assoc["twid_text"] = twid_text
    twid_assoc["twid_url"] = twid_url
    twid_assoc["twid_sn"] = twid_sn
    twid_assoc["sn_twid"] = sn_twid
    metadata = {}
    metadata["hashtag_sn"] = hashtag_sn
    metadata["sn_hashtag"] = sn_hashtag
    metadata["url_sn"] = url_sn
    metadata["sn_url"] = sn_url

    print("Processed " + str(count) + " tweets.")
    print("Found " + str(len(counters["users"])) + " users.")
    print("Found " + str(len(counters["hashtags"])) + " hashtags.")
    print("Found " + str(len(counters["urls"])) + " urls.")
    print("Found " + str(len(counters["amplifiers"])) + " amplifiers.")
    print("Found " + str(len(counters["influencers"])) + " influencers.")
    print("Found " + str(len(counters["repliers"])) + " repliers.")
    print("Found " + str(len(counters["replied_to"])) + " replied_to.")
    print("Found " + str(len(counters["retweeted"])) + " retweeted.")
    print("Found " + str(len(counters["retweeters"])) + " retweeters.")
    print("Found " + str(len(counters["mentioned"])) + " mentioned.")
    print("Found " + str(len(counters["mentioners"])) + " mentioners.")
    full = {}
    full["user_fields"] = user_fields
    full["counters"] = counters
    full["interactions"] = interactions
    full["twid_assoc"] = twid_assoc
    full["metadata"] = metadata
    return full

def get_rsn_sn(raw_data):
    sn_rsn = {}
    rsn_sn = {}
    for d in raw_data:
        sn = d["user"]["screen_name"]
        if "retweeted_status" in d:
            rsn = d["retweeted_status"]["user"]["screen_name"]
            if sn not in sn_rsn:
                sn_rsn[sn] = Counter()
            sn_rsn[sn][rsn] += 1
            if rsn not in rsn_sn:
                rsn_sn[rsn] = Counter()
            rsn_sn[rsn][sn] += 1
    return sn_rsn, rsn_sn

def get_twid_assoc(raw_data):
    twid_count = Counter()
    twid_rtc = Counter()
    twid_text = {}
    twid_url = {}
    twid_sn = {}
    sn_twid = {}
    for d in raw_data:
        sn = d["user"]["screen_name"]
        text = d["text"]
        twid = d["id_str"]
        rtc = d["retweet_count"]
        if "retweeted_status" in d:
            text = d["retweeted_status"]["text"]
            twid = d["retweeted_status"]["id_str"]
            rtc = d["retweeted_status"]["retweet_count"]
        text = text.replace("\n", " ")
        url = "https://twitter.com/" + sn + "/status/" + twid
        twid_count[twid] += 1
        twid_rtc[twid] = rtc
        twid_text[twid] = text
        twid_url[twid] = url
        if twid not in twid_sn:
            twid_sn[twid] = Counter()
        twid_sn[twid][sn] += 1
        if sn not in sn_twid:
            sn_twid[sn] = Counter()
        sn_twid[sn][twid] += 1
    return twid_count, twid_rtc, twid_text, twid_url, twid_sn, sn_twid

#####################################
# Timeline analysis
#####################################
def analyze_timeline_data(filename, keywords):
    retweets = Counter()
    users = Counter()
    user_tweet_count = Counter()
    user_retweet_count = Counter()
    who_retweeted_whom = {}
    who_retweeted_what = {}
    hashtags = Counter()
    hashtag_map = {}
    retweeted = Counter()
    twid_text = {}
    twid_url = {}
    interactions = {}
    keyword = []
    raw_data = make_timeline_iterator(filename)
    for item in raw_data:
        sn = list(item.keys())[0]
        tweets = list(item.values())[0]
        for d in tweets:
            if "verified" in d["user"] and d["user"]["verified"] == True:
                continue
            sn = d["user"]["screen_name"]
            if sn not in hashtag_map:
                hashtag_map[sn] = {}
            if "hashtags" in d:
                for ht in d["hashtags"]:
                    hashtags[ht] += 1
                    if ht not in hashtag_map[sn]:
                        hashtag_map[sn][ht] = 1
                    else:
                        hashtag_map[sn][ht] += 1
            if "retweeted_status" in d:
                r = d["retweeted_status"]
                user_retweet_count[sn] += 1
                if "verified" in r["user"] and r["user"]["verified"] == True:
                    continue
                if "retweet_count" in r and r["retweet_count"] < 1000:
                    continue
                rsn = r["user"]["screen_name"]
                if "hashtags" in r:
                    for ht in r["hashtags"]:
                        hashtags[ht] += 1
                        if ht not in hashtag_map[sn]:
                            hashtag_map[sn][ht] = 1
                        else:
                            hashtag_map[sn][ht] += 1
                retweeted[rsn] += 1
                users[rsn] += 1
                if sn not in interactions:
                    interactions[sn] = {}
                if rsn not in interactions[sn]:
                    interactions[sn][rsn] = 1
                else:
                    interactions[sn][rsn] += 1
                if sn not in who_retweeted_whom:
                    who_retweeted_whom[sn] = {}
                if rsn not in who_retweeted_whom[sn]:
                    who_retweeted_whom[sn][rsn] = 1
                else:
                    who_retweeted_whom[sn][rsn] += 1
                rtwid = r["id_str"]
                retweets[rtwid] += 1
                if rtwid not in who_retweeted_what:
                    who_retweeted_what[rtwid] = []
                if sn not in who_retweeted_what[rtwid]:
                    who_retweeted_what[rtwid].append(sn)
                rtext = r["text"].replace("\n", " ")
                if len(keywords) > 0:
                    for k in keywords:
                        if k in rtext:
                            if sn not in keyword:
                                keyword.append(sn)
                twid_text[rtwid] = rtext
                rurl = "https://twitter.com/" + rsn + "/status/" + rtwid
                twid_url[rtwid] = rurl
            else:
                user_tweet_count[sn] += 1
                text = d["text"].replace("\n", " ")
                if len(keywords) > 0:
                    for k in keywords:
                        if k in text:
                            if sn not in keyword:
                                keyword.append(sn)
                twid = d["id_str"]
                twid_text[twid] = text
                url = "https://twitter.com/" + sn + "/status/" + twid
                twid_url[twid] = url
    results = {}
    results["users"] = users
    results["hashtags"] = hashtags
    results["keyword"] = keyword
    results["hashtag_map"] = hashtag_map
    results["retweets"] = retweets
    results["retweeted"] = retweeted
    results["user_tweet_count"] = user_tweet_count
    results["user_retweet_count"] = user_retweet_count
    results["who_retweeted_whom"] = who_retweeted_whom
    results["who_retweeted_what"] = who_retweeted_what
    results["twid_text"] = twid_text
    results["twid_url"] = twid_url
    results["interactions"] = interactions
    return results

#####################################
# Searches that return user details
#####################################
def get_user_details(raw_data):
    details = {}
    for d in raw_data:
        sn = d["user"]["screen_name"]
        if sn not in details:
            details[sn] = d["user"]
        if "retweeted_status" in d:
            r = d["retweeted_status"]
            rsn = r["user"]["screen_name"]
            if rsn not in details:
                details[rsn] = r["user"]
    return details

#####################################
# Searches that return full data
#####################################
def get_data_from_tweets_snlist(raw_data, snlist):
    data = []
    for d in raw_data:
        sn = d["user"]["screen_name"]
        if sn in snlist:
            data.append(d)
    return data

def get_data_from_tweets_retweets_snlist(raw_data, snlist):
    data = []
    for d in raw_data:
        sn = d["user"]["screen_name"]
        if sn in snlist:
            data.append(d)
            continue
        if "retweeted_status" in d:
            rsn = d["retweeted_status"]["user"]["screen_name"]
            if rsn in snlist:
                data.append(d)
                continue
    return data

def get_data_from_interactions_snlist(raw_data, sn_list):
    all_data = []
    sn_list = [x.lower() for x in sn_list]
    for d in raw_data:
        sn = d["user"]["screen_name"].lower()
        matched = False
        if sn in sn_list:
            matched = True
        if matched == False and "interactions" in d:
            inter = [x.lower() for x in d["interactions"]]
            if len(set(inter).intersection(set(sn_list))) > 0:
                matched = True
        if matched == True:
            all_data.append(d)
    return all_data

def get_data_by_user_snlist(raw_data, snlist):
    tweets = {}
    for s in snlist:
        tweets[s] = []
    for d in raw_data:
        sn = d["user"]["screen_name"]
        if sn in snlist:
            tweets[sn].append(d)
    return tweets

def get_data_for_hashtags(raw_data, ht_list):
    all_data = []
    ht_list = [x.lower() for x in ht_list]
    for d in raw_data:
        if "hashtags" in d:
            ht = [x.lower() for x in d["hashtags"]]
            if len(set(ht).intersection(set(ht_list))) > 0:
                all_data.append(d)
    return all_data

def get_full_tweets_for_ids(raw_data, twid_list):
    id_to_tweet = {}
    for d in raw_data:
        twid = d["id_str"]
        if "retweeted_status" in d:
            twid = d["retweeted_status"]["id_str"]
        if twid in twid_list and twid not in id_to_tweet:
            id_to_tweet[twid] = d
    return id_to_tweet

def get_full_tweets_from_hashtags(raw_data, htlist):
    tweets = {}
    for h in htlist:
        tweets[h] = []
    for d in raw_data:
        if "hashtags" in d and d["hashtags"] is not None and len(d["hashtags"]) > 0:
            matched = list(set(d["hashtags"]).intersection(set(htlist)))
            if len(matched) > 0:
                for h in matched:
                    tweets[h].append(d)
    return tweets

#####################################
# Searches that return screen names
#####################################
def match_users_for_urls(raw_data, urls):
    users = {}
    for u in urls:
        users[u] = {}
    for d in raw_data:
        if "urls" in d and d["urls"] is not None and len(d["urls"]) > 0:
            for u in urls:
                if u in d["urls"]:
                    sn = d["user"]["screen_name"]
                    if sn not in users[u]:
                        users[u][sn] = 1
                    else:
                        users[u][sn] += 1
    return users

def match_users_for_hashtags(raw_data, hashtags):
    users = {}
    for h in hashtags:
        users[h] = {}
    for d in raw_data:
        if "hashtags" in d and d["hashtags"] is not None and len(d["hashtags"]) > 0:
            for h in hashtags:
                if h in d["hashtags"]:
                    sn = d["user"]["screen_name"]
                    if sn not in users[h]:
                        users[h][sn] = 1
                    else:
                        users[h][sn] += 1
    return users

def get_retweeted_from_sn_list(raw_data, sn_list):
    retweeted = []
    for d in raw_data:
        sn = d["user"]["screen_name"]
        if sn in sn_list:
            if "retweeted_status" in d:
                retweeted_sn = d["retweeted_status"]["user"]["screen_name"]
                if retweeted_sn not in retweeted:
                    retweeted.append(retweeted_sn)
    return retweeted

def get_retweeters_of_sn_list(raw_data, sn_list):
    retweeters = Counter()
    for d in raw_data:
        if "retweeted_status" in d:
            retweeted_sn = d["retweeted_status"]["user"]["screen_name"]
            if retweeted_sn in sn_list:
                sn = d["user"]["screen_name"]
                retweeters[sn] += 1
    return retweeters

def get_retweeters_per_sn(raw_data, sn_list):
    retweeters = {}
    for sn in sn_list:
        retweeters[sn] = Counter()
    for d in raw_data:
        if "retweeted_status" in d:
            retweeted_sn = d["retweeted_status"]["user"]["screen_name"]
            if retweeted_sn in sn_list:
                sn = d["user"]["screen_name"]
                retweeters[retweeted_sn][sn] += 1
    return retweeters

def match_descriptions(raw_data, match_words):
    matches = {}
    for m in match_words:
        matches[m] = []
    for d in raw_data:
        if "user" in d and "description" in d["user"]:
            desc = d["user"]["description"]
            if desc is not None:
                for m in match_words:
                    if find_exact_string(m.lower())(desc.lower()):
                        sn = d["user"]["screen_name"]
                        if sn not in matches[m]:
                            matches[m].append(sn)
    return matches

def get_retweeters_for_tweet_ids(raw_data, twid_list):
    retweeters = {}
    for d in raw_data:
        sn = d["user"]["screen_name"]
        twid = d["id_str"]
        if "retweeted_status" in d:
            twid = d["retweeted_status"]["id_str"]
        if twid in twid_list:
            if twid not in retweeters:
                retweeters[twid] = Counter()
            retweeters[twid][sn] += 1
    return retweeters

def get_sns_and_ids_from_data(raw_data):
    sns = Counter()
    sn_to_twid = {}
    for d in raw_data:
        sn = d["user"]["screen_name"]
        twid = d["user"]["id_str"]
        sns[sn] += 1
        sn_to_twid[sn] = twid
    return sns, sn_to_twid

def get_user_details_from_data(raw_data):
    details = []
    users = []
    for d in raw_data:
        sn = d["user"]["screen_name"]
        if sn not in users:
            users.append(sn)
            details.append(d["user"])
        if "retweeted_status" in d:
            rtsn = d["retweeted_status"]["user"]["screen_name"]
            if rtsn not in users:
                users.append(rtsn)
                details.append(d["retweeted_status"]["user"])
    return details

#####################################
# Searches that return tweet text
#####################################
def get_unique_tweets_from_data(raw_data):
    twid_counter = Counter()
    twid_url_map = {}
    twid_text_map = {}
    for d in raw_data:
        sn = None
        text = None
        twid = None
        if "retweeted_status" in d:
            s = d["retweeted_status"]
            text = s["text"].replace("\n", " ")
            twid = s["id_str"]
            sn = s["user"]["screen_name"]
        else:
            text = d["text"].replace("\n", " ")
            twid = d["id_str"]
            sn = d["user"]["screen_name"]
        url = "https://twitter.com/" + sn + "/status/" + twid
        twid_counter[twid] += 1
        twid_url_map[twid] = url
        twid_text_map[twid] = text
    tweets = Counter()
    tweet_url_map = {}
    for twid, count in twid_counter.most_common():
        tweets[twid_text_map[twid]] = count
        tweet_url_map[twid_text_map[twid]] = twid_url_map[twid]
    return tweets, tweet_url_map

def get_tweets_by_user(raw_data, userlist):
    tweets = {}
    tweet_url_map = {}
    twids = []
    for u in userlist:
        tweets[u] = []
    for d in raw_data:
        sn = d["user"]["screen_name"]
        twid = d["id_str"]
        text = d["text"].replace("\n", " ")
        url = "https://twitter.com/" + sn + "/status/" + twid
        if sn in userlist:
            if twid not in twids:
                if sn not in tweets:
                    tweets[sn] = []
                twids.append(twid)
                tweets[sn].append(text)
                tweet_url_map[text] = url
            if "retweeted_status" in d:
                s = d["retweeted_status"]
                text = s["text"].replace("\n", " ")
                twid = s["id_str"]
                sn = s["user"]["screen_name"]
                url = "https://twitter.com/" + sn + "/status/" + twid
                if twid not in twids:
                    if sn not in tweets:
                        tweets[sn] = []
                    twids.append(twid)
                    tweets[sn].append(text)
                    tweet_url_map[text] = url
    return tweets, tweet_url_map

def get_unique_tweets_from_snlist(raw_data, snlist):
    tweets = Counter()
    tweet_url_map = {}
    for d in raw_data:
        sn = d["user"]["screen_name"]
        twid = d["id_str"]
        text = d["text"].replace("\n", " ")
        url = "https://twitter.com/" + sn + "/status/" + twid
        if sn in snlist:
            tweet_url_map[text] = url
            tweets[text] += 1
            if "retweeted_status" in d:
                s = d["retweeted_status"]
                sn = s["user"]["screen_name"]
                twid = s["id_str"]
                text = s["text"].replace("\n", " ")
                url = "https://twitter.com/" + sn + "/status/" + twid
                tweet_url_map[text] = url
                tweets[text] += 1
    return tweets, tweet_url_map

def get_tweets_for_ids(raw_data, twid_list):
    id_to_tweet = {}
    for d in raw_data:
        text = d["text"]
        sn = d["user"]["screen_name"]
        twid = d["id_str"]
        if "retweeted_status" in d:
            twid = d["retweeted_status"]["id_str"]
            text = d["retweeted_status"]["text"]
            sn = d["retweeted_status"]["user"]["screen_name"]
        if twid in twid_list and twid not in id_to_tweet:
            text = text.replace("\n", " ")
            url = "https://twitter.com/" + sn + "/status/" + str(twid)
            id_to_tweet[twid] = [text, url]
    return id_to_tweet

#####################################
# Searches that return hashtags
#####################################
def get_hashtags_for_users(raw_data, userlist):
    hashtags = Counter()
    for d in raw_data:
        sn = d["user"]["screen_name"]
        if sn in userlist:
            for h in d["hashtags"]:
                hashtags[h] += 1
    return hashtags

#####################################
# Searches that return tweet ids
#####################################
def get_retweets_from_sn_list(raw_data, sn_list):
    retweet_ids = []
    retweeted = []
    for d in raw_data:
        sn = d["user"]["screen_name"]
        if sn in sn_list:
            if "retweeted_status" in d:
                rtwid = d["retweeted_status"]["id_str"]
                s = d["retweeted_status"]
                if "retweet_count" in s:
                    if s["retweet_count"] is not None:
                        if rtwid not in retweet_ids:
                            retweet_ids.append(rtwid)
                            retweeted.append(s)
    return retweeted

def get_unique_tweet_ids_from_snlist(raw_data, snlist):
    twids = Counter()
    for d in raw_data:
        if d["user"]["screen_name"] in snlist:
            twid = d["id_str"]
            if "retweeted_status" in d:
                twid = d["retweeted_status"]["id_str"]
            twids[twid] += 1
    return twids

def get_unique_tweet_ids_from_hashtags(raw_data, htlist):
    twids = Counter()
    for d in raw_data:
        if "hashtags" in d and d["hashtags"] is not None and len(d["hashtags"]) > 0:
            if len(list(set(d["hashtags"]).intersection(set(htlist)))) > 0:
                twid = d["id_str"]
                if "retweeted_status" in d:
                    twid = d["retweeted_status"]["id_str"]
                twids[twid] += 1
    return twids

def get_tweet_ids_from_hashtags(raw_data, htlist):
    twids = {}
    for ht in htlist:
        twids[ht] = Counter()
    for d in raw_data:
        if "hashtags" in d and d["hashtags"] is not None and len(d["hashtags"]) > 0:
            matched = list(set(d["hashtags"]).intersection(set(htlist)))
            if len(matched) > 0:
                twid = d["id_str"]
                if "retweeted_status" in d:
                    twid = d["retweeted_status"]["id_str"]
                for h in matched:
                    twids[h][twid] += 1
    return twids


