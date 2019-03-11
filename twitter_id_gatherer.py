# -*- coding: utf-8 -*-
from authentication_keys3 import *
from file_helpers import *
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy import API
from tweepy import Cursor

import json
import sys
import os
import io

def auth():
    acct_name, consumer_key, consumer_secret, access_token, access_token_secret = get_account_sequential()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    auth_api = API(auth)
    return auth_api

def get_statuses(id_list):
    batch_len = 100
    num_batches = int(len(id_list)/batch_len)
    batches = (id_list[i:i+batch_len] for i in range(0, len(id_list), batch_len))
    tweets = []
    for count, b in enumerate(batches):
        auth_api = auth()
        for id_str in b:
            status = None
            try:
                status = auth_api.get_status(id_str)
            except:
                pass
            if status is not None:
                tweets.append(status._json)
        msg = "Batch: " + str(count) + " Retrieved: " + str(len(tweets))
        sys.stdout.write(msg)
        sys.stdout.flush()
        sys.stdout.write("\r")
    return tweets

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Please supply path to list of tweet ids")
        sys.exit(0)

    id_list_fn = sys.argv[1]
    if not os.path.exists(id_list_fn):
        print("File: " + id_list_fn + " didn't exist.")
        sys.exit(0)
