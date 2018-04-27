from authentication_keys import *
from time_helpers import *
from process_tweet_object import *
from graph_helper import *
from process_text import *
from file_helpers import *

from TwitterAPI import TwitterAPI

import sys
import json
import os
import io
import re
import time

def get_auth():
    acct_name, consumer_key, consumer_secret, access_token, access_token_secret = get_account_sequential()
    auth = TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)
    return auth

def countdown_timer(val):
    countdown = val
    step = 1
    while countdown > 0:
        msg = "Time left: " + str(countdown)
        sys.stdout.write(msg)
        sys.stdout.flush()
        time.sleep(step)
        sys.stdout.write("\r")
        countdown -= step
    return

if __name__ == "__main__":
    targets = []
    if len(sys.argv) < 2:
        print("Please supply at least one screen_name to query")
        sys.exit(0)
    else:
        targets = sys.argv[1:]

    for t in targets:
        print("Getting all followers for: " + t)
        save_dir = os.path.join("captures/followers", t)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        cursor = -1
        count = 0

        follower_info = []
        while True:
            auth = get_auth()
            followers_raw = auth.request('followers/list', {'screen_name': t, 'count': 200, 'cursor':cursor, 'skip_status':True, 'include_user_entities': False})
            followers_clean = json.loads(followers_raw.response.text)
            filename = os.path.join(save_dir, "temp.json")
            save_json(followers_clean, filename)
            if "next_cursor" in followers_clean and followers_clean["next_cursor"] > 0:
                cursor = followers_clean['next_cursor']
                for follower in followers_clean["users"]:
                    entry = {}
                    fields = ["id_str", "name", "description", "screen_name", "followers_count", "friends_count", "statuses_count", "created_at", "favourites_count", "default_profile", "default_profile_image", "protected", "verified"]
                    for field in fields:
                        if field in follower and follower[field] is not None:
                            entry[field] = follower[field]
                    follower_info.append(entry)
            else:
                if "errors" in followers_clean:
                    for e in followers_clean["errors"]:
                        if "code" in e:
                            if e["code"] == 88:
                                print("Rate limit exceeded.")
                                countdown_timer(900)
                else:
                    print("Done. Found " + str(count) + " followers.")
                    break

            count += 200
            print("Queried: " + str(count))
            filename = os.path.join(save_dir, t)
            save_json(follower_info, filename)
            countdown_timer(15)
