from authentication_keys import *
from tweepy import OAuthHandler
from tweepy import API
from tweepy import Cursor
from datetime import datetime, date, time, timedelta
from collections import Counter
from process_tweet_object import *
from file_helpers import *
import sys
import re
import requests
import shutil

def dump_images(image_urls, dirname):
    for p in image_urls:
        m = re.search("^http:\/\/pbs\.twimg\.com\/media\/(.+)$", p)
        if m is not None:
            filename = m.group(1)
            print("Getting picture from: " + p)
            save_path = os.path.join(dirname, filename)
            response = requests.get(p, stream=True)
            with open(save_path, 'wb') as out_file:
                shutil.copyfileobj(response.raw, out_file)
            del response

acct_name, consumer_key, consumer_secret, access_token, access_token_secret = get_account_credentials() 

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
auth_api = API(auth)

account_list = []
if (len(sys.argv) > 1):
    account_list = sys.argv[1:]
else:
    print("Please provide a list of usernames at the command line.")
    sys.exit(0)

if len(account_list) > 0:
    for target in account_list:
        print("Getting data for " + target)
        save_dir = os.path.join("captures/likes/", target)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        all_tweets = []
        all_urls = []
        all_images = []
        tweet_count = 0
        end_date = datetime.utcnow() - timedelta(days=30)
        for status_obj in Cursor(auth_api.favorites, id=target).items():
            tweet_count += 1
            sys.stdout.write("\r")
            sys.stdout.flush()
            sys.stdout.write(str(tweet_count))
            sys.stdout.flush()
            status = status_obj._json
            urls = get_urls(status)
            for u in urls:
                if u not in all_urls:
                    all_urls.append(u)
            image_urls = get_image_urls(status)
            for u in image_urls:
                if u not in all_urls:
                    all_images.append(u)
            text = get_text(status)
            all_tweets.append(text)
        filename = os.path.join(save_dir, "tweets.json")
        save_json(all_tweets, filename)
        filename = os.path.join(save_dir, "urls.json")
        save_json(all_urls, filename)
        dirname = os.path.join(save_dir, "images")
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        dump_images(all_images, dirname)
