# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from authentication_keys import get_account_credentials
from time_helpers import *
from process_tweet_object import *
from graph_helper import *
from process_text import *
from file_helpers import *

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

##################
# Helper functions
##################
def debug_print(string):
    if debug == True:
        print string

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

def load_settings():
    debug_print(sys._getframe().f_code.co_name)
    global conf
    params_files = ["targets", "follow", "search", "languages"]
    for p in params_files:
        filename = "config/" + p + ".txt"
        conf[p] = read_config(filename)
    conf["params"] = {}
    conf["params"]["default_dump_interval"] = 10


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


def dump_event():
    debug_print(sys._getframe().f_code.co_name)
    if search == True:
        return
    if stopping == True:
        return
    global data, volume_file_handle, day_label, hour_label
    output = ""

# Dump text files
    prev_dump = get_counter("previous_dump_time")
    interval = get_counter("dump_interval")
    start_time = int(time.time())
    if start_time > prev_dump + interval:
        dump_counters()
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
        print
        print output

# Record tweet volumes
        volume_file_handle.write(current_time_str + "\t" + str("%.2f" % tcps) + "\n")

# Reload config
        load_settings()
        filename = os.path.join(save_dir, "conf.json")
        save_json(conf, filename)
        return

###############
# Process tweet
###############
def get_tweet_details(d):
    tweet_fields = ["id_str",
                    "text",
                    "lang",
                    "created_at", 
                    "retweet_count", 
                    "favorite_count", 
                    "quote_count", 
                    "reply_count", 
                    "source"]
    user_fields = ["id_str",
                   "screen_name",
                   "name",
                   "lang"
                   "friends_count",
                   "followers_count",
                   "description",
                   "statuses_count",
                   "favourites_count",
                   "listed_count"
                   "created_at",
                   "default_profile_image",
                   "default_profile",
                   "verified",
                   "protected"]
    entry = {}
    for f in tweet_fields:
        if f in d:
            entry[f] = d[f]
    if "retweeted_status" in d:
        s = d["retweeted_status"]
        entry["retweeted_status"] = {}
        for f in tweet_fields:
            if f in s:
                entry["retweeted_status"][f] = s[f]
        if "user" in s:
            u = s["user"]
            entry["retweeted_status"]["user"] = {}
            for f in user_fields:
                if f in u:
                    entry["retweeted_status"]["user"][f] = u[f]
    if "user" in d:
        u = d["user"]
        entry["user"] = {}
        for f in user_fields:
            if f in u:
                entry["user"][f] = u[f]
    return entry

def process_tweet(status):
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
    screen_name = user["screen_name"]
    text = get_text(status)
    if text is None:
        increment_counter("faulty_tweets")
        debug_print("Faulty tweet")
        return
    if "created_at" not in status:
        increment_counter("faulty_tweets")
        debug_print("Faulty tweet")
        return
    tweet_url = get_tweet_url(status)

    increment_counter("tweets_processed_this_interval")
    increment_counter("tweets_processed")

    abbreviated = get_tweet_details(status)
    dump_file_handle.write((unicode(json.dumps(abbreviated,ensure_ascii=False))) + u"\n")
    tweet_file_handle.write(unicode(screen_name) + u":\t" + unicode(text) + u"\n")
    tweet_url_file_handle.write(unicode(tweet_url) + u"\t" + unicode(text) + u"\n")

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
    mode = "stream"
    follow = False
    gardenhose = False
    save_dir = "data"
    input_params = []
    if len(sys.argv) > 1:
        for s in sys.argv[1:]:
            if s == "search":
                search = True
                mode = "search"
            elif s == "follow":
                follow = True
                mode = "follow"
            elif s == "stream":
                mode = "stream"
            elif s == "gardenhose":
                sample = True
                mode = "gardenhose"
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

    directories = [""]
    for d in directories:
        dirname = os.path.join(save_dir, d)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    load_settings()
    set_counter("dump_interval", conf["params"]["default_dump_interval"])
    set_counter("previous_dump_time", int(time.time()))
    set_counter("previous_graph_dump_time", int(time.time()))
    set_counter("script_start_time", int(time.time()))
    set_counter("previous_serialize", int(time.time()))
    set_counter("previous_config_reload", int(time.time()))

    tweet_file_handle = io.open(os.path.join(save_dir, "tweets.txt"), "a", encoding="utf-8")
    tweet_url_file_handle = io.open(os.path.join(save_dir, "tweet_urls.txt"), "a", encoding="utf-8")
    dump_file_handle = io.open(os.path.join(save_dir, "raw.json"), "a", encoding="utf-8")
    volume_file_handle = open(os.path.join(save_dir, "tweet_volumes.txt"), "a")

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
    filename = os.path.join(save_dir, "conf.json")
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
