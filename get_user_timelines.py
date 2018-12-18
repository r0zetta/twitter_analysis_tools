# -*- coding: utf-8 -*-
from authentication_keys import *
from process_tweet_object import *
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy import API
from tweepy import Cursor
import sys, io, os, json, time

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
    print("Account: " + acct_name)
    return auth_api

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

def get_tweets(sn):
    api = auth()
    tweets = []
    try:
        raw = api.user_timeline(screen_name = sn, count = 200, include_rts = True)
        raw = [x._json for x in raw]
        for r in raw:
            tweet = get_tweet_details(r)
            tweets.append(tweet)
    except:
        pass
    return tweets

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Please specify path to sn_list")
        sys.exit(0)

    sn_list_file = sys.argv[1]
    if not os.path.exists(sn_list_file):
        print("File: " + sn_list_file + " didn't exist.")
        sys.exit(0)

    sn_list = load_json(sn_list_file)
    base_dir = os.path.dirname(sn_list_file)
    save_file = os.path.join(base_dir, "timelines.json")
    print("Saving output to: " + save_file)
    print("")

    users = []
    if os.path.exists(save_file):
        print("Getting list of already queried sns")
        with io.open(save_file, "r", encoding="utf-8") as f:
            for line in file:
                entry = json.loads(line)
                sn, tweets = entry
                users.append(sn)

    print("Iterating through " + str(len(sn_list)) + " accounts.")

    with io.open(save_file, "a", encoding="utf-8") as f:
        for index, sn in enumerate(sn_list):
            if sn not in users:
                print("[" + str(index) + "] Getting tweets for " + sn)
                tweets = get_tweets(sn)
                entry = {}
                entry[sn] = tweets
                f.write(unicode(json.dumps(entry)) + u"\n")
