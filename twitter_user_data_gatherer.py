# -*- coding: utf-8 -*-
from time_helpers import *

from authentication_keys3 import get_account_credentials
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
    acct_name, consumer_key, consumer_secret, access_token, access_token_secret = get_account_credentials()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    auth_api = API(auth)
    return auth_api

def get_timeline(target):
    auth_api = auth()
    print("Signing in as: " + auth_api.me().name)
    print("Getting account details for " + target)
    objects = []
    count = 0
    for status_obj in Cursor(auth_api.user_timeline, id=target).items():
        status = status_obj._json
        objects.append(status)
        count += 1
        if count % 100 == 0:
            sys.stdout.write("\r")
            sys.stdout.flush()
            sys.stdout.write(str(count))
            sys.stdout.flush()
    print("")
    print("Done")
    return objects
