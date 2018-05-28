# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from authentication_keys import get_account_credentials
from time_helpers import *
from process_tweet_object import *
from graph_helper import *
from process_text import *
from file_helpers import *

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from nltk.cluster.util import cosine_distance
from collections import Counter
from itertools import combinations
from twarc import Twarc
from tweepy import OAuthHandler
from tweepy import API
from tweepy import Cursor
import numpy as np
import Queue
import threading
import sys
import time
import pickle
import os
import io
import re

##################
# Global variables
##################
stopping = False
debug = False
follow = False
search = False
tweet_queue = None
targets = []
to_follow = []
data = {}
conf = {}
stopwords = {}

############
# Clustering
############
def predict_tweet(tags):
    category = "UNK"
    if tags is not None:
        row = vectorize_item(tags, common_vocab)
        Y = np.array(row)
        Y = pca.transform(Y.reshape(1,-1))
        prediction = model.predict(Y)
        category = "%02d" % prediction[0]
    return category

##################
# Helper functions
##################
def debug_print(string):
    if debug == True:
        print string

def write_daily_data(name, output_fn, prefix, suffix, label):
    if day_label in data:
        if name in data[day_label]:
            dirname = prefix + "daily/" + name
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            filename = prefix + "daily/" + name + "/" + name + "_" + day_label + suffix
            output_fn(data[day_label][name], filename)

def write_hourly_data(name, output_fn, prefix, suffix, label):
    if hour_label in data:
        if name in data[hour_label]:
            dirname = prefix + "hourly/" + name
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            filename = prefix + "hourly/" + name + "/" + name + "_" + hour_label + suffix
            output_fn(data[hour_label][name], filename)

def save_output(name, filetype):
    output_fn = None
    suffix = ""
    if "json" in filetype:
        output_fn = save_json
        prefix = os.path.join(save_dir, "json/")
        suffix = ".json"
    elif "csv" in filetype:
        output_fn = save_counter_csv
        prefix = os.path.join(save_dir, "")
        suffix = ".csv"
    elif "gephi" in filetype:
        output_fn = save_gephi_csv
        prefix = os.path.join(save_dir, "")
        suffix = ".csv"

    if name in data:
        filename = prefix + "overall/" + name + suffix
        output_fn(data[name], filename)
    if search == False:
        write_daily_data(name, output_fn, prefix, suffix, day_label)
        write_hourly_data(name, output_fn, prefix, suffix, hour_label)
    else:
        day_labels = []
        hour_labels = []
        for key, vals in data.iteritems():
            m = re.search("^([0-9]+)$", key)
            if m is not None:
                captured = m.group(1)
                if len(captured) == 8:
                    day_labels.append(captured)
                else:
                    hour_labels.append(captured)
        for l in day_labels:
            write_daily_data(name, output_fn, prefix, suffix, l)
        for l in hour_labels:
            write_daily_data(name, output_fn, prefix, suffix, l)


def cleanup():
    debug_print(sys._getframe().f_code.co_name)
    global dump_file_handle, volume_file_handle, tweet_file_handle, tweet_url_file_handle, stopping
    script_end_time_str = time.strftime("%Y-%m-%d %H:%M:%S")
    conf["last_stopped"] = script_end_time_str
    if len(threading.enumerate()) > 2:
        print "Waiting for queue to empty..."
        stopping = True
        tweet_queue.join()
    time.sleep(2)
    tweet_file_handle.close()
    tweet_url_file_handle.close()
    dump_file_handle.close()
    volume_file_handle.close()
    print "Serializing data..."
    serialize()
    dump_data()
    dump_graphs()

def load_settings():
    debug_print(sys._getframe().f_code.co_name)
    global conf
    if "settings" not in conf:
        conf["settings"] = {}
    conf["settings"] = read_settings("config/settings.txt")

    params_files = ["monitored_hashtags", "targets", "follow", "search", "ignore", "keywords", "good_users", "bad_users", "url_keywords", "languages", "description_keywords", "legit_sources", "fake_news_sources"]
    for p in params_files:
        filename = "config/" + p + ".txt"
        conf[p] = read_config(filename)
    conf["params"] = {}
    conf["params"]["default_dump_interval"] = 10
    conf["params"]["serialization_interval"] = 180
    conf["params"]["graph_dump_interval"] = 60


##########
# Storage
##########
def check_for_counter(name):
    debug_print(sys._getframe().f_code.co_name)
    debug_print(name)
    global data
    if "counters" not in data:
        data["counters"] = {}
    if name not in data["counters"]:
        data["counters"][name] = 0

def increment_counter(name):
    debug_print(sys._getframe().f_code.co_name)
    debug_print(name)
    global data
    check_for_counter(name)
    data["counters"][name] += 1

def set_counter(name, value):
    debug_print(sys._getframe().f_code.co_name)
    global data
    check_for_counter(name)
    data["counters"][name] = value

def get_counter(name):
    debug_print(sys._getframe().f_code.co_name)
    check_for_counter(name)
    return data["counters"][name]

def get_all_counters():
    debug_print(sys._getframe().f_code.co_name)
    if "counters" in data:
        return data["counters"]

def record_user_details(user):
    debug_print(sys._getframe().f_code.co_name)
    global data
    if "user_details" not in data:
        data["user_details"] = {}
    id_str = user_get_id_str(user)
    if id_str is not None and len(id_str) > 0:
        user_data = get_user_details_dict(user)
        data["user_details"][id_str] = user_data

def record_list(label, item):
    debug_print(sys._getframe().f_code.co_name)
    global data
    if label not in data:
        data[label] = []
    if item not in data[label]:
        data[label].append(item)

def record_freq_dist(label, item, collect_periodic=True):
    debug_print(sys._getframe().f_code.co_name)
    global data
    if label not in data:
        data[label] = Counter()
    data[label][item] += 1

    if collect_periodic == True:
        if tweet_day_label not in data:
            data[tweet_day_label] = {}
        if label not in data[tweet_day_label]:
            data[tweet_day_label][label] = Counter()
        data[tweet_day_label][label][item] += 1

        if tweet_hour_label not in data:
            data[tweet_hour_label] = {}
        if label not in data[tweet_hour_label]:
            data[tweet_hour_label][label] = Counter()
        data[tweet_hour_label][label][item] += 1

def record_map(label, key, value, collect_periodic=True):
    debug_print(sys._getframe().f_code.co_name)
    global data
    if label not in data:
        data[label] = {}
    if key not in data[label]:
        data[label][key] = []
    if value not in data[label][key]:
        data[label][key].append(value)

    if collect_periodic == True:
        if tweet_day_label not in data:
            data[tweet_day_label] = {}
        if label not in data[tweet_day_label]:
            data[tweet_day_label][label] = {}
        if key not in data[tweet_day_label][label]:
            data[tweet_day_label][label][key] = []
        if value not in data[tweet_day_label][label][key]:
            data[tweet_day_label][label][key].append(value)

        if tweet_hour_label not in data:
            data[tweet_hour_label] = {}
        if label not in data[tweet_hour_label]:
            data[tweet_hour_label][label] = {}
        if key not in data[tweet_hour_label][label]:
            data[tweet_hour_label][label][key] = []
        if value not in data[hour_label][label][key]:
            data[tweet_hour_label][label][key].append(value)

def record_freq_dist_map(label, item1, item2, collect_periodic=True):
    debug_print(sys._getframe().f_code.co_name)
    global data
    if label not in data:
        data[label] = {}
    if item1 not in data[label]:
        data[label][item1] = Counter()
    data[label][item1][item2] += 1

    if collect_periodic == True:
        if tweet_day_label not in data:
            data[tweet_day_label] = {}
        if label not in data[tweet_day_label]:
            data[tweet_day_label][label] = {}
        if item1 not in data[tweet_day_label][label]:
            data[tweet_day_label][label][item1] = Counter()
        data[tweet_day_label][label][item1][item2] += 1

        if tweet_hour_label not in data:
            data[tweet_hour_label] = {}
        if label not in data[tweet_hour_label]:
            data[tweet_hour_label][label] = {}
        if item1 not in data[tweet_hour_label][label]:
            data[tweet_hour_label][label][item1] = Counter()
        data[tweet_hour_label][label][item1][item2] += 1

def record_interarrival(name, tweet_time):
    global data
    debug_print(sys._getframe().f_code.co_name)
    if "interarrivals" not in data:
        data["interarrivals"] = {}
    if name in data["interarrivals"]:
        inter = data["interarrivals"][name]
        if "previous_tweeted" in inter:
            delta = tweet_time - inter["previous_tweeted"]
            if delta > 0:
                if delta not in data["interarrivals"][name]:
                    data["interarrivals"][name][delta] = 1
                else:
                    data["interarrivals"][name][delta] += 1
    else:
        data["interarrivals"][name] = {}
    data["interarrivals"][name]["previous_tweeted"] = tweet_time

def calculate_interarrival_statistics(name):
    debug_print(sys._getframe().f_code.co_name)
    stdev = 0.0
    counts = []
    if "interarrivals" in data and name in data["interarrivals"]:
        inter = data["interarrivals"][name]
        for key, val in inter.iteritems():
            if key != "previous_tweeted":
                counts.append(val)
        if len(counts) > 0:
            stdev = float(np.std(counts))
    return stdev

def get_network_params():
    debug_print(sys._getframe().f_code.co_name)
    edges = 0
    nodes = 0
    if "user_user_map" in data:
        edges = sum([len(x) for x in data["user_user_map"].values()])
        nodeset = set()
        for source, targets in data["user_user_map"].iteritems():
            nodeset.add(source)
            for target, value in targets.iteritems():
                nodeset.add(target)
        nodes = len(nodeset)
    return nodes, edges

######################
# Follow functionality
######################
def get_ids_from_names(names):
    print("Got " + str(len(names)) + " names.")
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    auth_api = API(auth)

    batch_len = 100
    batches = (names[i:i+batch_len] for i in range(0, len(names), batch_len))
    all_json = []
    for batch_count, batch in enumerate(batches):
        users_list = auth_api.lookup_users(screen_names=batch)
        users_json = (map(lambda t: t._json, users_list))
        all_json += users_json

    ret = []
    found_names = []
    for d in all_json:
        if "id_str" in d:
            id_str = d["id_str"]
            ret.append(id_str)
        if "screen_name" in d:
            found_names.append(d["screen_name"])
    not_found = list(set([x.lower() for x in names]) - set([x.lower() for x in found_names]))
    return ret, not_found

#############
# Dump graphs
#############
def record_volume_data(category, label, timestamp, value):
    debug_print(sys._getframe().f_code.co_name)
    global data
    if category not in data:
        data[category] = {}
    if label not in data[category]:
        data[category][label] = []
    data[category][label].append([timestamp, value])

def get_volume_labels(category):
    debug_print(sys._getframe().f_code.co_name)
    global data
    ret = []
    if category in data:
        for label, stuff in data[category].iteritems():
            ret.append(label)
    return ret

def get_volume_data(category, label):
    debug_print(sys._getframe().f_code.co_name)
    global data
    ret = {}
    if category in data:
        if label in data[category]:
            ret = data[category][label]
    return ret

def dump_tweet_volume_graphs():
    debug_print(sys._getframe().f_code.co_name)
    labels = get_volume_labels("tweet_volumes")
    for l in labels:
        volume_data = get_volume_data("tweet_volumes", l)
        if len(volume_data) > 5:
            dates = []
            volumes = []
            for item in volume_data:
                dates.append(item[0])
                volumes.append(item[1])
            chart_data = {}
            chart_data["tweets/sec"] = volumes
            dirname = os.path.join(save_dir, "")
            filename = "_tweet_volumes_" + l + ".svg"
            title = "Tweet Volumes (" + l + ")"
            dump_line_chart(dirname, filename, title, dates, chart_data)

def dump_languages_graphs():
    debug_print(sys._getframe().f_code.co_name)
    counter_data = get_all_counters()
    prefixes = ["tweets", "captured_tweets"]
    for p in prefixes:
        if counter_data is not None:
            chart_data = {}
            for name, value in sorted(counter_data.iteritems(), key=lambda x:x[1], reverse= True):
                m = re.search("^" + p + "_([a-z][a-z][a-z]?)$", name)
                if m is not None:
                    item = m.group(1)
                    chart_data[item] = value
            dirname = os.path.join(save_dir, "")
            filename = "_" + p + "_lang_breakdown.svg"
            title = "Language breakdown"
            dump_pie_chart(dirname, filename, title, chart_data)

def create_overall_graphs(name):
    debug_print(sys._getframe().f_code.co_name)
    if name is None:
        return
    if name in data:
        dataset = dict(data[name].most_common(20))
        if len(dataset) > 0:
            title = name
            dirname = os.path.join(save_dir, "graphs/overall/" + name + "/pie/")
            filename = name + ".svg"
            dump_pie_chart(dirname, filename, title, dataset)
            dirname = os.path.join(save_dir, "graphs/overall/" + name + "/bar/")
            x_labels = []
            dump_bar_chart(dirname, filename, title, x_labels, dataset)

def create_periodic_graphs(name):
    debug_print(sys._getframe().f_code.co_name)
    if name is None:
        return
    for x, y in {"hour": "hourly", "day": "daily"}.iteritems():
        file_label = get_datestring(x)
        title = name + " " + file_label
# Create pie chart for latest datestring
        if file_label in data:
            if name in data[file_label]:
                dataset = dict(data[file_label][name].most_common(15))
                if len(dataset) > 0:
                    dirname = os.path.join(save_dir, "graphs/" + y + "/" + name + "/pie/")
                    filename = name + "_" + file_label + ".svg"
                    dump_pie_chart(dirname, filename, title, dataset)

# Get all items from last 10 datestrings
        all_items = Counter()
        for s in range(10):
            label = get_datestring(x, s)
            if label in data:
                if name in data[label]:
                    for n, v in data[label][name].most_common(20):
                        if n not in all_items:
                            all_items[n] += v
# Create bar chart for latest datestring
        chart_data = {}
        x_labels = []
        for item, val in all_items.most_common(20):
            chart_data[item] = []
        for s in list(reversed(range(10))):
            label = get_datestring(x, s)
            if label in data and name in data[label]:
                dataset = dict(data[label][name].most_common(20))
                if len(dataset) > 0:
                    x_labels.append(label)
                    for item in chart_data.keys():
                        if item in dataset.keys():
                            chart_data[item].append(dataset[item])
                        else:
                            chart_data[item].append(0)

        dirname = os.path.join(save_dir, "graphs/" + y + "/" + name + "/bar/")
        filename = name + "_" + file_label + ".svg"
        #x_labels = list(reversed(x_labels))
        dump_bar_chart(dirname, filename, title, x_labels, chart_data)

########################
# Periodically dump data
########################

def dump_counters():
    debug_print(sys._getframe().f_code.co_name)
    counter_dump = get_all_counters()
    val_output = ""
    date_output = ""
    if counter_dump is not None:
        for n, c in sorted(counter_dump.iteritems()):
            val = None
            if type(c) is float:
                val = "%.2f"%c
                val_output += unicode(val) + u"\t" + unicode(n) + u"\n"
            elif len(str(c)) > 9:
                val = unix_time_to_readable(int(c))
                date_output += unicode(val) + u"\t" + unicode(n) + u"\n"
            else:
                val = c
                val_output += unicode(val) + u"\t" + unicode(n) + u"\n"
    handle = io.open(os.path.join(save_dir, "_counters.txt"), "w", encoding='utf-8')
    handle.write(unicode(val_output))
    handle.write(u"\n")
    handle.write(unicode(date_output))
    handle.close

def serialize():
    debug_print(sys._getframe().f_code.co_name)
    filename = os.path.join(save_dir, "raw/serialized.bin")
    if os.path.exists(filename):
        tmp_file = os.path.join(save_dir, "raw/serialized.bak")
        os.rename(filename, tmp_file)
    save_bin(data, filename)

# These get dumped when we exit
    print("Performing extended serialization.")
    filename = os.path.join(save_dir, "raw/conf.json")
    save_json(conf, filename)

    raw = ["interarrivals", "tag_map", "who_tweeted_what", "who_tweeted_what_url", "who_retweeted_what", "who_retweeted_what_url", "user_user_map", "user_hashtag_map", "user_cluster_map", "sources", "user_details"]
    for n in raw:
        if n in data:
            filename = os.path.join(save_dir, "raw/" + n + ".json")
            save_json(data[n], filename)

    jsons = ["all_users", "all_hashtags", "influencers", "amplifiers", "word_frequencies", "all_urls", "urls_not_twitter", "fake_news_urls", "fake_news_tweeters", "suspiciousness_scores"]
    for n in jsons:
        save_output(n, "json")
    return

def dump_data():
    debug_print(sys._getframe().f_code.co_name)
    dump_counters()
    dump_languages_graphs()
    dump_tweet_volume_graphs()

    csvs = ["amplifiers", "influencers", "all_users", "all_hashtags", "influencers", "word_frequencies", "all_urls", "urls_not_twitter", "fake_news_urls", "fake_news_tweeters", "cluster_counts"]
    for n in csvs:
        save_output(n, "csv")

    gephis = ["user_user_map", "user_hashtag_map", "user_cluster_map"]
    for n in gephis:
        save_output(n, "gephi")

    if "suspiciousness_scores" in data:
        filename = os.path.join(save_dir, "custom/most_suspicious.csv")
        save_counter_csv(data["suspiciousness_scores"], filename)

    custom = ["description_matches", "keyword_matches", "hashtag_matches", "url_matches", "interarrival_matches", "interacted_with_bad", "interacted_with_suspicious", "suspicious_users", "interesting_clusters_user", "interesting_clusters_hashtag", "interesting_clusters_keyword", "interesting_clusters_url", "high_frequency"]
    for n in custom:
        if n in data:
            filename = os.path.join(save_dir, "custom/" + n + ".json")
            save_json(data[n], filename)

    return

def dump_graphs():
    debug_print(sys._getframe().f_code.co_name)
    graphs = ["all_users", "all_hashtags", "influencers", "amplifiers", "word_frequencies", "cluster_counts"]
    for g in graphs:
        create_periodic_graphs(g)
        create_overall_graphs(g)
    return

def dump_event():
    debug_print(sys._getframe().f_code.co_name)
    if search == True:
        return
    if stopping == True:
        return
    global data, volume_file_handle, day_label, hour_label
    output = ""

# Dump text files
    interval = get_counter("dump_interval")
    prev_dump = get_counter("previous_dump_time")
    start_time = int(time.time())
    if start_time > prev_dump + interval:
        dump_data()
        dump_time = int(time.time()) - start_time
        output += "Data dump took: " + str(dump_time) + " seconds.\n"

# Dump graphs
        interval = get_counter("graph_dump_interval")
        prev_dump = get_counter("previous_graph_dump_time")
        start_time = int(time.time())
        if start_time > prev_dump + interval:
            dump_graphs()
            set_counter("previous_graph_dump_time", int(time.time()))
            dump_time = int(time.time()) - start_time
            output += "Graph dump took: " + str(dump_time) + " seconds.\n"

# Serialize
        interval = get_counter("serialization_interval")
        prev_dump = get_counter("previous_serialize")
        start_time = int(time.time())
        if start_time > prev_dump + interval:
            set_counter("previous_serialize", int(time.time()))
            serialize()
            dump_time = int(time.time()) - start_time
            output += "Serialization took: " + str(dump_time) + " seconds.\n"

        current_time = int(time.time())
        processing_time = current_time - get_counter("previous_dump_time")

        queue_length = tweet_queue.qsize()
        output += str(queue_length) + " items in the queue.\n"

        tweets_seen = get_counter("tweets_processed_this_interval")
        output += "Processed " + str(tweets_seen) + " tweets during the last " + str(processing_time) + " seconds.\n"
        tweets_captured = get_counter("tweets_captured_this_interval")
        output += "Captured " + str(tweets_captured) + " tweets during the last " + str(processing_time) + " seconds.\n"

        output += "Tweets encountered: " + str(get_counter("tweets_encountered")) + ", captured: " + str(get_counter("tweets_captured")) + ", processed: " + str(get_counter("tweets_processed")) + "\n"

        tpps = float(float(get_counter("tweets_processed_this_interval"))/float(processing_time))
        set_counter("tweets_per_second_processed_this_interval", tpps)
        output += "Processed/sec: " + str("%.2f" % tpps) + "\n"

        tcps = float(float(get_counter("tweets_captured_this_interval"))/float(processing_time))
        set_counter("tweets_per_second_captured_this_interval", tcps)
        output += "Captured/sec: " + str("%.2f" % tcps) + "\n"

        for key in ["all_users", "all_hashtags", "all_urls", "influencers", "amplifiers", "suspicious_users", "interacted_with_bad", "interacted_with_suspicious", "keyword_matches", "hashtag_matches", "description_matches", "url_matches"]:
            if key in data:
                val = len(data[key])
                set_counter(key, val)
                output += key + ": " + str(val) + "\n"

        nodes, edges = get_network_params()
        output += "Nodes: " + str(nodes) + " Edges: " + str(edges) + "\n"

        set_counter("tweets_processed_this_interval", 0)
        set_counter("tweets_captured_this_interval", 0)
        set_counter("processing_time", processing_time)
        increment_counter("successful_loops")
        output += "Executed " + str(get_counter("successful_loops")) + " successful loops.\n"

        total_running_time = int(time.time()) - get_counter("script_start_time")
        set_counter("total_running_time", total_running_time)
        output += "Running as " + acct_name + " since " + script_start_time_str + " (" + str(total_running_time) + " seconds)\n"

        current_time_str = time.strftime("%Y-%m-%d %H:%M:%S")
        output += "Current time is: " + current_time_str + "\n\n"

        set_counter("average_tweets_per_second", tcps)
        set_counter("previous_dump_time", int(time.time()))
        #os.system('clear')
        print
        print output

# Record tweet volumes
        record_volume_data("tweet_volumes", "all_tweets", current_time_str, tcps)
        volume_file_handle.write(current_time_str + "\t" + str("%.2f" % tcps) + "\n")

# Reload config
        load_settings()
        filename = os.path.join(save_dir, "raw/conf.json")
        save_json(conf, filename)

# Update timestamp labels and delete old data
        day_label = get_datestring("day")
        if day_label not in data:
            data[day_label] = {}
        hour_label = get_datestring("hour")
        if hour_label not in data:
            data[hour_label] = {}
        for offset in range(10, 100):
            offset_label = get_datestring("day", offset)
            if offset_label in data:
                del(data[offset_label])
            offset_label = get_datestring("hour", offset)
            if offset_label in data:
                del(data[offset_label])
        return

###############
# Process tweet
###############
def process_tweet(status):
    global data, tweet_hour_label, tweet_day_label
    debug_print(sys._getframe().f_code.co_name)
# If the tweet doesn't contain a user object or "text" it's useless
    if "user" not in status:
        increment_counter("faulty_tweets")
        debug_print("Faulty tweet")
        return
    user = status["user"]
    if "screen_name" not in user:
        increment_counter("faulty_tweets")
        debug_print("Faulty tweet")
        return
    text = get_text(status)
    if text is None:
        increment_counter("faulty_tweets")
        debug_print("Faulty tweet")
        return
    if "created_at" not in status:
        increment_counter("faulty_tweets")
        debug_print("Faulty tweet")
        return
# At this point, we're good to process the tweet
    increment_counter("tweets_processed_this_interval")
    increment_counter("tweets_processed")

    susp_score = 0
    ignore_list = conf["ignore"]
    bad_users = conf["bad_users"]
    good_users = conf["good_users"]
    monitored_hashtags = conf["monitored_hashtags"]
    keywords = conf["keywords"]
    description_keywords = conf["description_keywords"]
    fake_news_sources = conf["fake_news_sources"]
    url_keywords = conf["url_keywords"]

    created_at = status["created_at"]
    screen_name = user["screen_name"]
    record_freq_dist("all_users", screen_name)
    tweet_id = status["id_str"]
    user_id = user["id_str"]
    lang = status["lang"]
    text = text.strip()
    text = re.sub("\n", " ", text)

    record_user_details(user)
    if is_egg(status):
        susp_score += 100

    source = get_tweet_source(status)
    if source is not None:
        record_freq_dist_map("sources", screen_name, source, False)

    account_age_days = get_account_age_days(status)
    tweets_per_day = get_tweets_per_day(status)
    if tweets_per_day > 100:
        record_list("high_frequency", screen_name)
    if account_age_days < 30:
        if tweets_per_day > 100:
            susp_score += (tweets_per_day - 100) * (30 - account_age_days)
    else:
        susp_score += tweets_per_day - 100

    followers_count = get_followers_count(status)
    if followers_count < 100:
        if tweets_per_day > 100:
            susp_score += tweets_per_day - 100

    if account_age_days > 30 and followers_count < 5:
        susp_score += 100

    friends_count = get_friends_count(status)
    follow_ratio = 0
    if friends_count > 0 and followers_count > 0:
        follow_ratio = float(friends_count)/float(followers_count)
    if follow_ratio < 1.2:
        susp_score += (follow_ratio * 5)

# Create some useable time formats
    tweet_time_object = twitter_time_to_object(created_at)
    tweet_time_unix = twitter_time_to_unix(created_at)
    tweet_hour_label = tweet_time_object.strftime("%Y%m%d%H")
    tweet_day_label = tweet_time_object.strftime("%Y%m%d")
    tweet_url = get_tweet_url(status)

    record_interarrival(screen_name, tweet_time_unix)
    interarrival_stdev = calculate_interarrival_statistics(screen_name)
    if interarrival_stdev > 0:
        susp_score += interarrival_stdev * 10
        if "interarrival_matches" not in data:
            data["interarrival_matches"] = {}
        data["interarrival_matches"][screen_name] = interarrival_stdev

# Dump raw status to json
    if "dump_raw_data" in conf["settings"]:
        if conf["settings"]["dump_raw_data"] == True:
            dump_file_handle.write((unicode(json.dumps(status,ensure_ascii=False))) + u"\n")

# Dump tweet to disk
    if "record_all_tweets" in conf["settings"]:
        if conf["settings"]["record_all_tweets"] == True:
            tweet_file_handle.write(unicode(screen_name) + u":\t" + unicode(text) + u"\n")
            tweet_url_file_handle.write(unicode(tweet_url) + u"\t" + unicode(text) + u"\n")

# Process text, record who tweeted what, which cluster the tweets belongs to, and build tag map
    debug_print("Preprocess text")
    preprocessed = preprocess_text(text, lang)
    cluster = None
    if "tag_map" not in data:
        data["tag_map"] = {}
    if preprocessed is not None:
        record_map("who_tweeted_what", preprocessed, screen_name, False)
        record_map("who_tweeted_what_url", screen_name, tweet_url, False)
        tags = []
        if preprocessed not in data["tag_map"]:
            if lang in nlp and lang in stemmer:
                debug_print("Processing with spacy")
                tags = process_sentence_nlp(preprocessed, lang, nlp[lang], stemmer[lang])
            elif stopwords is not None and lang in stopwords:
                debug_print("Tokenizing with stopwords")
                tags = tokenize_sentence(preprocessed, stopwords[lang])
            else:
                debug_print("Tokenizing without stopwords")
                tags = tokenize_sentence(preprocessed)
            if tags is not None and len(tags) > 0:
                debug_print("Adding tags to tag map")
                data["tag_map"][preprocessed] = tags
        else:
            tags = data["tag_map"][preprocessed]
        if tags is not None and len(tags) > 0:
            for t in tags:
                record_freq_dist("word_frequencies", t)
            if clustering_enabled == True:
                cluster = predict_tweet(tags)

    if cluster is not None:
        record_freq_dist("cluster_counts", cluster)
        record_freq_dist_map("user_cluster_map", cluster, screen_name, False)

    retweeted_user = get_retweeted_user(status)
    if retweeted_user is not None and preprocessed is not None:
        retweeted_status = get_retweeted_status(status)
        if retweeted_status is not None:
            record_map("who_retweeted_what", retweeted_status, screen_name, False)
        retweeted_url = get_retweeted_tweet_url(status)
        if retweeted_url is not None:
            record_map("who_retweeted_what_url", retweeted_url, screen_name, False)

# Check text for keywords
    matched = False
    for k in keywords:
        if k.lower() in text.lower():
            susp_score += 100
            matched = True
    if matched == True:
        record_list("keyword_matches", screen_name)
        if cluster is not None:
            record_list("interesting_clusters_keyword", cluster)

# Process hashtags
    debug_print("Process hashtags")
    hashtags = get_hashtags(status)
    if len(hashtags) > 0:
        matched = False
        for h in hashtags:
            if h.lower() in [x.lower() for x in monitored_hashtags]:
                susp_score += 100
                matched = True
            record_freq_dist("all_hashtags", h)
            if cluster is not None:
                record_freq_dist_map("hashtag_cluster_map", cluster, h, False)
        if screen_name.lower() not in [x.lower() for x in ignore_list]:
            for h in hashtags:
                record_freq_dist_map("user_hashtag_map", screen_name.lower(), h.lower(), False)
        if matched == True:
            record_list("hashtag_matches", screen_name)
            if cluster is not None:
                record_list("interesting_clusters_hashtag", cluster)

# Process URLs
    debug_print("Process URLs")
    urls = get_urls(status)
    tweeted_fake_news = False
    if len(urls) > 0:
        for u in urls:
            record_freq_dist("all_urls", u)
            if "twitter" not in u:
                record_freq_dist("urls_not_twitter", u)
            matched = False
            for k in url_keywords:
                if k in u:
                    susp_score += 100
                    matched = True
            if matched == True:
                record_list("url_matches", u)
            for f in fake_news_sources:
                if f in u:
                    tweeted_fake_news = True
                    record_freq_dist("fake_news_urls", u)
                    susp_score += 100
    if tweeted_fake_news == True:
        record_freq_dist("fake_news_tweeters", screen_name)
        if cluster is not None:
            record_list("interesting_clusters_url", cluster)

# Process interactions
    debug_print("Process interactions")
    interactions = get_interactions(status)
    if len(interactions) > 0:
        for n in interactions:
            record_freq_dist("influencers", n)
            record_freq_dist("amplifiers", screen_name)
        if screen_name.lower() not in [x.lower() for x in ignore_list]:
            for n in interactions:
                if n.lower() not in [x.lower() for x in ignore_list]:
                    record_freq_dist_map("user_user_map", screen_name.lower(), n.lower(), False)
                matched_bad = False
                matched_susp = False
                if screen_name.lower() not in [x.lower() for x in good_users]:
                    for u in bad_users:
                        if u.lower() == n.lower():
                            susp_score += 100
                            matched_bad = True
                    if "suspicious_users" in data:
                        for u in data["suspicious_users"]:
                            if u.lower() == n.lower():
                                susp_score += 100
                                matched_susp = True
                if matched_bad == True:
                    record_list("interacted_with_bad", screen_name)
                    if cluster is not None:
                        record_list("interesting_clusters_user", cluster)
                if matched_susp == True:
                    record_list("interacted_with_suspicious", screen_name)
                    if cluster is not None:
                        record_list("interesting_clusters_user", cluster)

    if screen_name.lower() in [x.lower() for x in bad_users]:
        if cluster is not None:
            record_list("interesting_clusters_user", cluster)

    if is_bot_name(screen_name):
        susp_score += 100

# Processing description
    if "description" in user:
        description = user["description"]
        if description is None:
            susp_score += 100
        else:
            description = description.lower()
            if len(description) < 1:
                susp_score += 100
            matched = False
            for k in description_keywords:
                if k.lower() in description:
                    matched = True
                    susp_score += 100
            if matched == True:
                record_list("description_matches", screen_name)


# Known users get a zero suspiciousness
    if screen_name.lower() in [x.lower() for x in good_users]:
        susp_score = 0

    if susp_score > 900:
        record_list("suspicious_users", screen_name)

    if "suspiciousness_scores" not in data:
        data["suspiciousness_scores"] = Counter()
    data["suspiciousness_scores"][screen_name] = int(susp_score)

    debug_print("Done processing")
    return

def preprocess_tweet(status):
    debug_print(sys._getframe().f_code.co_name)
    increment_counter("tweets_encountered")
    debug_print("Preprocessing status")
    if status is None:
        debug_print("No status")
        sys.stdout.write("-")
        sys.stdout.flush()
        return
    if "lang" not in status:
        debug_print("No lang")
        sys.stdout.write("-")
        sys.stdout.flush()
        return
    lang = status["lang"]
    debug_print("lang="+lang)
    increment_counter("tweets_" + lang)
    if len(conf["languages"]) > 0:
        if lang not in conf["languages"]:
            debug_print("Skipping tweet of lang: " + lang)
            sys.stdout.write("-")
            sys.stdout.flush()
            return
    increment_counter("captured_tweets_" + lang)
    increment_counter("tweets_captured")
    increment_counter("tweets_captured_this_interval")
    tweet_queue.put(status)
    sys.stdout.write("#")
    sys.stdout.flush()

def tweet_processing_thread():
    debug_print(sys._getframe().f_code.co_name)
    while True:
        item = tweet_queue.get()
        process_tweet(item)
        dump_event()
        tweet_queue.task_done()
    return

def start_thread():
    debug_print(sys._getframe().f_code.co_name)
    global tweet_queue
    print "Starting processing thread..."
    tweet_queue = Queue.Queue()
    t = threading.Thread(target=tweet_processing_thread)
    t.daemon = True
    t.start()
    return

def get_tweet_stream(query):
    debug_print(sys._getframe().f_code.co_name)
    if follow == True:
        for tweet in t.filter(follow=query):
            preprocess_tweet(tweet)
    elif search == True:
        for tweet in t.search(query):
            preprocess_tweet(tweet)
    else:
        if query == "":
            for tweet in t.sample():
                preprocess_tweet(tweet)
        else:
            for tweet in t.filter(track=query):
                preprocess_tweet(tweet)

#########################################
# Main routine, called when script starts
#########################################
if __name__ == '__main__':
    follow = False
    save_dir = "data"
    input_params = []
    if len(sys.argv) > 1:
        for s in sys.argv[1:]:
            if s == "search":
                search = True
            elif s == "follow":
                follow = True
            elif s == "debug":
                debug = True
            else:
                input_params.append(s)

    if search == True:
        unixtime = get_utc_unix_time()
        save_dir = os.path.join("captures/searches", str(int(unixtime)))
    if search == True and follow == True:
        print("Only one of search and follow params can be supplied")
        sys.exit(0)

    directories = ["",
                   "custom",
                   "raw",
                   "json",
                   "json/overall",
                   "json/daily",
                   "json/hourly",
                   "graphs",
                   "graphs/overall",
                   "graphs/daily",
                   "graphs/hourly",
                   "hourly",
                   "daily",
                   "overall"]
    for d in directories:
        dirname = os.path.join(save_dir, d)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    if os.path.exists("config/stopwords.json"):
        stopwords = load_json("config/stopwords.json")

# Deserialize from previous run
    data = {}
    if search == False:
        filename = os.path.join(save_dir, "raw/serialized.bin")
        if os.path.exists(filename):
            print("Attempting to deserialize from " + filename)
            old_data = load_bin(filename)
            if old_data is not None:
                data = old_data
            else:
                tmp_file = os.path.join(save_dir, "raw/serialized.bak")
                if os.path.exists(tmp_file):
                    print("Serialized data was corrupted. Using backup file from " + tmp_file)
                    tmp_data = load_bin(tmp_file)
                    if tmp_data is None:
                        print("Backup file was also corrupted. Deserialization failed.")
                        sys.exit(0)
                    else:
                        data = tmp_data

    tweet_day_label = ""
    tweet_hour_label = ""
    day_label = get_datestring("day")
    if day_label not in data:
        data[day_label] = {}
    hour_label = get_datestring("hour")
    if hour_label not in data:
        data[hour_label] = {}

    load_settings()
    set_counter("dump_interval", conf["params"]["default_dump_interval"])
    set_counter("serialization_interval", conf["params"]["serialization_interval"])
    set_counter("graph_dump_interval", conf["params"]["graph_dump_interval"])
    set_counter("previous_dump_time", int(time.time()))
    set_counter("previous_graph_dump_time", int(time.time()))
    set_counter("script_start_time", int(time.time()))
    set_counter("previous_serialize", int(time.time()))
    set_counter("previous_config_reload", int(time.time()))

    tweet_file_handle = io.open(os.path.join(save_dir, "raw/tweets.txt"), "a", encoding="utf-8")
    tweet_url_file_handle = io.open(os.path.join(save_dir, "raw/tweet_urls.txt"), "a", encoding="utf-8")
    dump_file_handle = io.open(os.path.join(save_dir, "raw/raw.json"), "a", encoding="utf-8")
    volume_file_handle = open(os.path.join(save_dir, "raw/tweet_volumes.txt"), "a")

# Init spacy and stemmer
    print("Languages: " + ", ".join(conf["languages"]))
    langs = conf["languages"]
    nlp, stemmer = init_nlp_multi_lang(langs)

# Init clustering model, if available
    cluster_dir = "clusters"
    clustering_enabled = False
    model_file = os.path.join(cluster_dir, "k_means_model.sav")
    pca_file = os.path.join(cluster_dir, "pca_model.sav")
    vocab_file = os.path.join(cluster_dir, "common_vocab.json")
    pca = None
    model = None
    common_vocab = None
    print("Attempting to load clustering model")
    if os.path.exists(model_file) and os.path.exists(pca_file) and os.path.exists(vocab_file):
        pca = PCA()
        pca = pickle.load(open(pca_file, "rb"))
        model = KMeans()
        model = pickle.load(open(model_file, "rb"))
        common_vocab = load_json(vocab_file)
        clustering_enabled = True
        print("Clustering enabled")
    else:
        print("Clustering disabled")

# Initialize twitter object
    acct_name, consumer_key, consumer_secret, access_token, access_token_secret = get_account_credentials()
    t = Twarc(consumer_key, consumer_secret, access_token, access_token_secret)
    print "Signing in as: " + acct_name

# Determine mode and build query
    query = ""
    if follow == True:
        print("Listening to accounts")
        conf["mode"] = "follow"
        to_follow = []
        if len(input_params) > 0:
            to_follow = input_params
            conf["input"] = "command_line"
        else:
            conf["input"] = "config_file"
            to_follow = read_config("config/follow.txt")
            to_follow = [x.lower() for x in to_follow]
        if len(to_follow) < 1:
            print("No account names provided.")
            sys.exit(0)
        print("Converting names to IDs")
        print("Names to follow: " + str(len(to_follow)))
        id_list, not_found = get_ids_from_names(to_follow)
        print("ID count: " + str(len(id_list)))
        if len(not_found) > 0:
            print("Not found: " + ", ".join(not_found))
        if len(id_list) < 1:
            print("No account IDs found.")
            sys.exit(0)
        query = ",".join(id_list)
        conf["query"] = query
        conf["ids"] = id_list
        print "Preparing stream"
        print "IDs: " + query
    elif search == True:
        conf["mode"] = "search"
        print("Performing Twitter search")
        searches = []
        if len(input_params) > 0:
            searches = input_params
            conf["input"] = "command_line"
        else:
            conf["input"] = "config_file"
            searches = read_config("config/search.txt")
            searches = [x.lower() for x in searches]
        if len(searches) < 1:
            print("No search terms supplied.")
            sys.exit(0)
        if len(searches) > 1:
            print("Search can only handle one search term (for now).")
            sys.exit(0)
        query = searches[0]
        conf["query"] = query
        print "Preparing search"
        print "Query: " + query
    else:
        conf["mode"] = "stream"
        print("Listening to Twitter search stream with targets:")
        targets = []
        if len(input_params) > 0:
            targets = input_params
            conf["input"] = "command_line"
        else:
            conf["input"] = "config_file"
            targets = read_config("config/targets.txt")
        if len(targets) > 0:
            query = ",".join(targets)
            conf["query"] = query
            print "Preparing stream"
            if query == "":
                print "Getting 1% sample."
            else:
                print "Search: " + query
        print targets

# Start a thread to process incoming tweets
    start_thread()

# Start stream
    script_start_time_str = time.strftime("%Y-%m-%d %H:%M:%S")
    conf["first_started"] = script_start_time_str
    filename = os.path.join(save_dir, "raw/conf.json")
    save_json(conf, filename)
    if search == True:
        set_counter("successful_loops", 0)
        try:
            get_tweet_stream(query)
        except KeyboardInterrupt:
            print "Keyboard interrupt..."
            cleanup()
            sys.exit(0)
        except:
            print
            print "Something exploded..."
            cleanup()
            sys.exit(0)
        cleanup()
        sys.exit(0)
    else:
        while True:
            set_counter("successful_loops", 0)
            try:
                get_tweet_stream(query)
            except KeyboardInterrupt:
                print "Keyboard interrupt..."
                cleanup()
                sys.exit(0)
            except:
                print
                print "Something exploded..."
                cleanup()
                sys.exit(0)

# ToDo:
# Search mode, multiple params - different save dirs, or overlapping
# Retweet spikes - also in search mode
# sources
# hashtag hashtag interactions
