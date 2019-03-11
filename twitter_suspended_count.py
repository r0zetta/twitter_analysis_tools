# -*- coding: utf-8 -*-
from authentication_keys3 import *

from collections import Counter
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy import API
from tweepy import Cursor
from tweepy import TweepError

import json
import time
import sys
import re
import os
import io

def load_json(fn):
    ret = None
    with io.open(fn, "r", encoding="utf-8") as f:
        ret = json.load(f)
    return ret

def save_json(d, fn):
    with open(fn, "w") as f:
        f.write(json.dumps(d, indent=4))

def auth():
    acct_name, consumer_key, consumer_secret, access_token, access_token_secret = get_account_sequential()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    auth_api = API(auth)
    return auth_api

def get_users(sn_list, save_file):
    print("Save file: " + save_file)
    users = []
    for count, sn in enumerate(sn_list):
        if count % 100 == 0 and count > 0:
            save_json(users, save_file)
        q = get_user(sn)
        print(str(count) + str(q))
        users.append(q)
    return users

def get_user(sn):
    auth_api = auth()
    ret = "valid"
    user = None
    try:
        user = auth_api.get_user(sn)
    except TweepError as e:
        if "suspended" in e.reason:
            ret = "suspended"
        elif "not found" in e.reason:
            ret = "unknown"
        pass
    return [sn, ret]

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Please supply path to list of users or ids")
        sys.exit(0)

    sn_list_fn = sys.argv[1]
    if not os.path.exists(sn_list_fn):
        print("File: " + sn_list_fn + " didn't exist.")
        sys.exit(0)

    base_dir = os.path.dirname(sn_list_fn)
    save_file = os.path.join(base_dir, "suspended_analysis.json")

    sn_list = load_json(sn_list_fn)
    users = get_users(sn_list, save_file)
    save_json(users, save_file)
