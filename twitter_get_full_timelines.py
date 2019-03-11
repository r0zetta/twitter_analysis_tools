# -*- coding: utf-8 -*-
from authentication_keys import *
from process_tweet_object import *

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

def load_json(filename):
    ret = None
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                ret = json.load(f)
        except:
            pass
    return ret

def get_tweet_details(d):
    tweet_fields = ["id_str",
                    "text",
                    "lang",
                    "created_at", 
                    "in_reply_to_screen_name",
                    "in_reply_to_status_id",
                    "is_quote_status",
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
                   "location",
                   "statuses_count",
                   "favourites_count",
                   "listed_count"
                   "created_at",
                   "default_profile_image",
                   "default_profile",
                   "verified",
                   "protected"]
    entry = {}
    entry["hashtags"] = get_hashtags_preserve_case(d)
    entry["urls"] = get_urls(d)
    entry["interactions"] = get_interactions_preserve_case(d)
    entry["retweeted"] = get_retweeted_user(d)
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
    save_file = os.path.join(base_dir, "full_timelines.json")
    print("Saving output to: " + save_file)
    print("")

    users = []
    queried_file = os.path.join(base_dir, "full_timelines_queried.json")
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
        print("")
        if target in users:
            print("Already queried: " + target)
            continue
        auth_api = auth()
        print("Signing in as: " + auth_api.me().name)
        print("Getting tweets for account: " + target)
        details = {}
        details[target] = []
        count = 0
        try:
            for status_obj in Cursor(auth_api.user_timeline, id=target).items():
                count += 1
                status = status_obj._json
                entry = get_tweet_details(status)
                details[target].append(entry)
                sys.stdout.write("\r")
                sys.stdout.flush()
                sys.stdout.write(str(count))
                sys.stdout.flush()
            f.write(json.dumps(details) + "\n")
            qf.write(target + "\n")
        except:
            pass
    f.close()
    qf.close()







