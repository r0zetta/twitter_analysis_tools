# -*- coding: utf-8 -*-
from authentication_keys import *
from time_helpers import *

from collections import Counter
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy import API
from tweepy import Cursor

from datetime import datetime, date, time, timedelta
import numpy as np
import os.path
import json
import time
import sys
import re
import os
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

def auth():
    acct_name, consumer_key, consumer_secret, access_token, access_token_secret = get_account_sequential()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    auth_api = API(auth)
    return auth_api

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Please specify path to sn_list")
        sys.exit(0)

    sn_list_file = sys.argv[1]
    print("sn_list: " + sn_list_file)
    if not os.path.exists(sn_list_file):
        print("File: " + sn_list_file + " didn't exist.")
        sys.exit(0)

    sn_list = load_json(sn_list_file)
    print("Accounts to query: " + str(len(sn_list)))
    base_dir = os.path.dirname(sn_list_file)
    save_file = os.path.join(base_dir, "heatmap_analysis.json")
    print("Saving output to: " + save_file)
    print("")

    users = []
    queried_file = os.path.join(base_dir, "heatmap_queried.json")
    if os.path.exists(queried_file):
        print("Getting list of already queried sns")
        with open(queried_file, "r") as f:
            for line in f:
                users.append(line.strip())
    print("Already queried " + str(len(users)) + " users")

    sn_len = len(sn_list)
    print("Iterating through " + str(sn_len) + " accounts.")

    qf = open(queried_file, "a")
    f = open(save_file, "a")

    for target in sn_list:
        if target in users:
            print("Already queried: " + target)
            continue
        auth_api = auth()
        print "Signing in as: " + auth_api.me().name
        previous_tweet_time = None
        print("Getting tweets for account: " + target)
        tweet_timestamps = []
        timestamps = []
        interarrivals = {}
        count = 0
        try:
            for status_obj in Cursor(auth_api.user_timeline, id=target).items():
                count += 1
                status = status_obj._json
                ts = status["created_at"]
                timestamps.append(ts)
                tweet_time = status_obj.created_at
                tweet_timestamps.append(tweet_time)
                if previous_tweet_time is not None:
                    delta = previous_tweet_time - tweet_time
                    delta_seconds = int(delta.total_seconds())
                    if delta_seconds not in interarrivals:
                        interarrivals[delta_seconds] = 1
                    else:
                        interarrivals[delta_seconds] += 1
                previous_tweet_time = tweet_time
                sys.stdout.write("\r")
                sys.stdout.flush()
                sys.stdout.write(str(count))
                sys.stdout.flush()

            std = np.std(interarrivals.values())
            heatmap = create_heatmap(tweet_timestamps)
            entry = {}
            entry["screen_name"] = target
            entry["tweets_queried"] = count
            entry["timestamps"] = timestamps
            entry["interarrivals"] = interarrivals
            entry["interarrival_stdev"] = std
            entry["heatmap"] = heatmap
            f.write(json.dumps(entry) + "\n")
            qf.write(target + "\n")
        except:
            pass
    f.close()
    qf.close()







