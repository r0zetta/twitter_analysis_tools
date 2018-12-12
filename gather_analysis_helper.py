import sys, os, io, random, json, re
import pandas as pd
from collections import Counter
from time_helpers import *

report_every = 100000

def find_exact_string(w):
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

def load_json(fn):
    ret = None
    with io.open(fn, "r", encoding="utf-8") as f:
        ret = json.load(f)
    return ret

def save_json(fn, d):
    with io.open(fn, "w", encoding="utf-8") as f:
        f.write(json.dumps(d, indent=4))

def save_csv(inter, fn):
    with io.open(fn, "w", encoding="utf-8") as f:
        f.write("Source,Target,Weight\n")
        for source, targets in inter.items():
            for target, count in targets.items():
                f.write(source + "," + target + "," + str(count) + "\n")

def print_sn_counter(sn_list, num):
    for user, count in sn_list.most_common(num):
        snstr = "https://twitter.com/" + user
        print(str(count) + "\t" + snstr)

def print_hashtag_counter(ht_list, num):
    for ht, count in ht_list.most_common(num):
        htstr = "https://twitter.com/search?q=%23" + ht
        print(str(count) + "\t" + htstr)

def print_counter(ct, num):
    for item, count in ct.most_common(num):
        print(str(count) + "\t" + item)

def tokenize_sentence(text, stopwords):
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
    for token in tokens:
        if len(token) > 0:
            if stopwords is not None:
                if token in stopwords:
                    token = None
            if token is not None:
                if re.search(".+…$", token):
                    token = None
            if token is not None:
                if token == "#":
                    token = None
            if token is not None:
                if token[-1] == ".":
                    token = token[:-1]
            if token is not None:
                cleaned.append(token)
    if len(cleaned) < 1:
        return []
    return cleaned

def read_from_raw_data(start_time, end_time):
    ret = []
    print("Reading data from: " + start_time + " to: " + end_time)
    start = time_object_to_unix(time_string_to_object(start_time))
    end = time_object_to_unix(time_string_to_object(end_time))
    count = 0
    with io.open("data/raw.json", "r", encoding="utf-8") as f:
        for line in f:
            count += 1
            if count % report_every == 0:
                print("Count: " + str(count))
            d = json.loads(line)
            timestamp = twitter_time_to_unix(d["created_at"])
            if timestamp >= start and timestamp <= end:
                ret.append(d)
    print("Saw a total of " + str(count) + " records.")
    print(str(len(ret)) + " records matched the date range.")
    return ret

def create_cache(start_time, end_time, dirname):
    filename = str(int(time.time())) + ".cache"
    cachefile = os.path.join(dirname, filename)
    print("Creating cache with data from: " + self.start_time + " to: " + self.end_time)
    start = time_object_to_unix(time_string_to_object(start_time))
    end = time_object_to_unix(time_string_to_object(end_time))
    cf = io.open(cachefile, "a", encoding="utf-8")
    count = 0
    matched = 0
    with io.open("data/raw.json", "r", encoding="utf-8") as f:
        for line in f:
            count += 1
            if count % report_every == 0:
                print("Count: " + str(count))
            d = json.loads(line)
            timestamp = twitter_time_to_unix(d["created_at"])
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
            if timestamp >= start and timestamp <= end:
                yield(d)

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

def get_full_tweets_by_user(raw_data, userlist):
    tweets = {}
    for u in userlist:
        tweets[u] = []
    for d in raw_data:
        sn = d["user"]["screen_name"]
        if sn in userlist:
            tweets[sn].append(d)
    return tweets

def get_tweets_by_user(raw_data, userlist):
    tweets = {}
    for u in userlist:
        tweets[u] = []
    for d in raw_data:
        sn = d["user"]["screen_name"]
        if sn in userlist:
            text = d["text"]
            tweets[sn].append(text)
    return tweets

def get_unique_tweets_from_snlist(raw_data, snlist):
    tweets = Counter()
    for d in raw_data:
        if d["user"]["screen_name"] in snlist:
            tweets[d["text"]] += 1
    return tweets

def get_hashtags_for_users(raw_data, userlist):
    hashtags = Counter()
    for d in raw_data:
        sn = d["user"]["screen_name"]
        if sn in userlist:
            for h in d["hashtags"]:
                hashtags[h] += 1
    return hashtags

def match_graph(inter, match_list):
    ret = {}
    for source, targets in inter.items():
        if source in match_list:
            ret[source] = targets
            continue
        for target, weight in targets.items():
            if target in match_list:
                ret[source] = targets
                break
    return ret

def match_graph2(inter, match_list):
    ret = {}
    for source, targets in inter.items():
        if source in match_list:
            for target, weight in targets.items():
                if target in match_list:
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
    return user_fields, counters, interactions

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
            counters["highly_retweeted_users"][retweeted_sn] += 1
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

def get_data_for_snlist(raw_data, sn_list):
    all_data = []
    sn_list = [x.lower() for x in sn_list]
    for d in raw_data:
        sn = d["user"]["screen_name"].lower()
        matched = False
        if sn in sn_list:
            matched = True
        if "interactions" in d:
            inter = [x.lower for x in d["interactions"]]
            if len(set(inter).intersection(set(sn_list))) > 0:
                matched = True
        if matched == True:
            all_data.append(d)
    return all_data

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

def get_full_tweets_from_snlist(raw_data, snlist):
    tweets = {}
    for s in snlist:
        tweets[s] = []
    for d in raw_data:
        sn = d["user"]["screen_name"]
        if sn in snlist:
            tweets[sn].append(d)
    return tweets

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



