# -*- coding: utf-8 -*-
from authentication_keys import get_account_credentials
from time_helpers import *
from process_text import *
from process_tweet_object import *
from graph_helper import *
from file_helpers import *

from collections import Counter
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy import API
from tweepy import Cursor

from datetime import datetime, date, time, timedelta
import numpy as np
import pygal
import os.path
import requests
import shutil
import json
import time
import sys
import re
import os
import io

# Add a way to figure out if the account "never sleeps"
# Total up the columns in heatmap. If the total is less than a specified amount, consider that "sleep"


def get_user_objects(follower_ids):
    batch_len = 100
    num_batches = len(follower_ids) / 100
    batches = (follower_ids[i:i+batch_len] for i in range(0, len(follower_ids), batch_len))
    all_data = []
    for batch_count, batch in enumerate(batches):
        sys.stdout.write("\r")
        sys.stdout.flush()
        sys.stdout.write("Fetching batch: " + str(batch_count) + "/" + str(num_batches))
        sys.stdout.flush()
        users_list = auth_api.lookup_users(user_ids=batch)
        users_json = (map(lambda t: t._json, users_list))
        all_data += users_json
    return all_data

def output_data(num=None):
    outputs = ["hashtags", "urls", "retweeted", "replied", "interacted", "languages", "words", "sources"]
    output_string = ""
    for o in outputs:
        if o in data:
            output_string += u"\n" + o + u":\n\n"
            for item, count in Counter(data[o]).most_common(num):
                output_string += unicode(count) + u": " + unicode(item) + u"\n"
    return output_string

def dump_chronology(dirname):
    global data
    types = ["per_hour", "per_day", "per_week", "per_month"]
    labels = ["all_tweets"]
    for t in types:
        for l in labels:
            x_axis_labels = []
            plot_data = {}
            title = t + "_" + l
            y_axis_label = "tweets/minute"
            filename = title + ".svg"
            if title in data:
                plot_data[l] = []
                seen = 0
                for name, count in sorted(data[title].items(), key=lambda x:x[0], reverse=False):
                    x_axis_labels.append(name)
                    plot_data[l].append(count)
                    seen += 1
                if seen > 100:
                    dump_line_chart(dirname, filename, title, x_axis_labels, plot_data)
                else:
                    dump_bar_chart(dirname, filename, title, x_axis_labels, plot_data)

def record_chronology(label, tweet_time):
    global data
    timestamps = {}
    timestamps["per_hour"] = time_object_to_hour(tweet_time)
    timestamps["per_day"] = time_object_to_day(tweet_time)
    timestamps["per_week"] = time_object_to_week(tweet_time)
    timestamps["per_month"] = time_object_to_month(tweet_time)
    types = ["per_hour", "per_day", "per_week", "per_month"]
    for t in types:
        l = t + "_" + label
        if l not in data:
            data[l] = {}
        if timestamps[t] not in data[l]:
            data[l][timestamps[t]] = 1
        else:
            data[l][timestamps[t]] += 1

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

# Creates one week length ranges and finds items that fit into those range boundaries
def make_ranges(user_data, num_ranges=20):
    range_max = 604800 * num_ranges
    range_step = range_max/num_ranges
    ranges = {}
    labels = {}
    for x in range(num_ranges):
        start_range = x * range_step
        end_range = x * range_step + range_step
        label = "%02d" % x + " - " + "%02d" % (x+1) + " weeks"
        labels[label] = []
        ranges[label] = {}
        ranges[label]["start"] = start_range
        ranges[label]["end"] = end_range
    for user in user_data:
        if "created_at" in user:
            account_age = seconds_since_twitter_time(user["created_at"])
            for label, timestamps in ranges.iteritems():
                if account_age > timestamps["start"] and account_age < timestamps["end"]:
                    entry = {}
                    id_str = user["id_str"]
                    entry[id_str] = {}
                    fields = ["screen_name", "name", "created_at", "friends_count", "followers_count", "favourites_count", "statuses_count"]
                    for f in fields:
                        if f in user:
                            entry[id_str][f] = user[f]
                    labels[label].append(entry)
    return labels

def analyze_account_creation_dates(dataset, dirname, prefix):
    timestamps = []
    print("Getting timestamps")
    for entry in dataset:
        if "created_at" in entry:
            timestamp = twitter_time_to_unix(entry["created_at"])
            timestamps.append(timestamp)
    timestamps = sorted(timestamps, reverse=True)
    interarrivals = Counter()
    deltas = []
    previous_timestamp = 0
    for t in timestamps:
        if previous_timestamp == 0:
            previous_timestamp = t
            continue
        delta = previous_timestamp - t
        interarrivals[delta] += 1
        deltas.append(delta)
        previous_timestamp = t

    filename = os.path.join(dirname, prefix + "_account_creation_interarrivals.csv")
    with open(filename, "w") as f:
        for val, count in interarrivals.most_common():
            f.write(str(val) + "," + str(count) + "\n")
    filename = os.path.join(dirname, prefix + "_account_creation_deltas.csv")
    with open(filename, "w") as f:
        for val in deltas[:3000]:
            f.write(str(val) + "\n")

def create_dist_graph(distdata, name):
    dirname = save_dir
    filename = name + ".svg"
    title = name
    x_labels = []
    dataset = []
    for val, count in sorted(distdata.items()):
        x_labels.append(val)
        dataset.append(count)
    chart_data = {}
    chart_data[title] = dataset
    dump_bar_chart(dirname, filename, title, x_labels, chart_data)

def auth():
    acct_name, consumer_key, consumer_secret, access_token, access_token_secret = get_account_credentials()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    auth_api = API(auth)
    return auth_api

if __name__ == '__main__':
    base_dir = "captures/users/"
    save_images = False
    targets = []
    if (len(sys.argv) > 1):
        for a in sys.argv[1:]:
            if "save_images" in a:
                save_images = True
            else:
                targets.append(a)

    if len(targets) < 1:
        print("No targets specified.")
        sys.exit(0)
    print "Targets: " + ", ".join(targets)


    stopwords = load_json("config/stopwords.json")

    for target in targets:
        auth_api = auth()
        print "Signing in as: " + auth_api.me().name
        print("Getting account details for " + target)
        user = auth_api.get_user(target)
        if user is None:
            continue
        save_dir = os.path.join(base_dir, target)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        data = {}
        data["name"] = user.name
        data["screen_name"] = user.screen_name
        data["user_id"] = user.id_str
        data["statuses_count"] = user.statuses_count
        data["favourites_count"] = user.favourites_count
        data["listed_count"] = user.listed_count
        data["friends_count"] = user.friends_count
        data["followers_count"] = user.followers_count
        data["description"] = user.description
        account_created_date = user.created_at
        data["account_created_date_readable"] = time_object_to_string(account_created_date)
        delta = datetime.utcnow() - account_created_date
        data["account_age_days"] = delta.days
        data["tweets_per_day"] = 0
        data["tweets_per_hour"] = 0
        if data["account_age_days"] > 0:
            data["tweets_per_day"] = float(data["statuses_count"])/float(data["account_age_days"])
            data["tweets_per_hour"] = float(data["statuses_count"])/float(data["account_age_days"] * 24)

        data["tweet_count"] = 0
        data["original_tweet_count"] = 0
        data["retweet_count"] = 0
        data["quote_count"] = 0
        data["reply_count"] = 0
        tweet_timestamps = []
        data["hashtags"] = Counter()
        data["urls"] = Counter()
        data["sources"] = Counter()
        image_urls = set()
        tweet_texts = set()
        data["interarrivals"] = {}
        data["interactions"] = {}
        data["interacted"] = Counter()
        data["retweeted"] = Counter()
        data["quoted"] = Counter()
        data["replied"] = Counter()
        data["languages"] = Counter()
        data["words"] = Counter()
        previous_tweet_time = None
        print("Getting tweets for account: " + target)
        max_s = 3200
        if data["statuses_count"] < 3200:
            max_s = data["statuses_count"]
        retweet_dist = {}
        like_dist = {}
        total_likes = 0
        most_liked = 0
        most_liked_tweet = ""
        total_retweets = 0
        most_retweeted = 0
        most_retweeted_tweet = ""
        own_tweets = 0
        for status_obj in Cursor(auth_api.user_timeline, id=target).items():
            data["tweet_count"] += 1
            status = status_obj._json

            text = get_text(status)
            text = text.strip()
            text = text.replace('\n', ' ').replace('\r', '')

            tweet_time = status_obj.created_at
            record_chronology("all_tweets", tweet_time)
            date_string = time_object_to_string(tweet_time)
            tweet_timestamps.append(tweet_time)
            if previous_tweet_time is not None:
                delta = previous_tweet_time - tweet_time
                delta_seconds = int(delta.total_seconds())
                if delta_seconds not in data["interarrivals"]:
                    data["interarrivals"][delta_seconds] = 1
                else:
                    data["interarrivals"][delta_seconds] += 1
            previous_tweet_time = tweet_time
            inter = get_interactions(status)
            if len(inter) > 0:
                if data["screen_name"] not in data["interactions"]:
                    data["interactions"][data["screen_name"]] = Counter()
                for n in inter:
                    if n != data["screen_name"]:
                        data["interacted"][n] += 1
                        data["interactions"][data["screen_name"]][n] += 1
            own_tweet = True
            if get_retweeted_status(status) is not None:
                data["retweeted"][get_retweeted_user(status)] += 1
                data["retweet_count"] += 1
                own_tweet = False
            if get_quoted(status) is not None:
                data["quoted"][get_quoted(status)] += 1
                data["quote_count"] += 1
            if get_replied(status) is not None:
                data["replied"][get_replied(status)] += 1
                data["reply_count"] += 1
            if own_tweet == True:
                own_tweets += 1
                data["original_tweet_count"] += 1
                num_likes = status["favorite_count"]
                if num_likes > 0:
                    total_likes += num_likes
                    if num_likes > most_liked:
                        most_liked = num_likes
                        most_liked_tweet = text
                    if num_likes not in like_dist:
                        like_dist[num_likes] = 1
                    else:
                        like_dist[num_likes] += 1
                num_retweets = status["retweet_count"]
                if num_retweets > 0:
                    total_retweets += num_retweets
                    if num_retweets > most_retweeted:
                        most_retweeted = num_retweets
                        most_retweeted_tweet = text
                    if num_retweets not in retweet_dist:
                        retweet_dist[num_retweets] = 1
                    else:
                        retweet_dist[num_retweets] += 1

            ht = get_hashtags(status)
            if ht is not None and len(ht) > 0:
                for h in ht:
                    data["hashtags"][h] += 1
            ur = get_urls(status)
            if ur is not None and len(ur) > 0:
                for u in ur:
                    data["urls"][u] += 1
            imu = get_image_urls(status)
            if imu is not None and len(imu) > 0:
                for m in imu:
                    image_urls.add(m)
            data["sources"][status["source"]] += 1
            data["languages"][status["lang"]] += 1

            tweet_texts.add(text)
            prep = preprocess_text(text)
            if prep is not None and len(prep) > 0:
                sw = None
                if status["lang"] in stopwords:
                    sw = stopwords[status["lang"]]
                tokens = tokenize_sentence(prep, sw)
                if tokens is not None and len(tokens) > 0:
                    for t in tokens:
                        data["words"][t] += 1
            sys.stdout.write("\r")
            sys.stdout.flush()
            sys.stdout.write(str(data["tweet_count"]) + "/" + str(max_s))
            sys.stdout.flush()

        save_json(data["retweeted"], os.path.join(save_dir, "retweeted.json"))
        mean_likes = np.mean(like_dist.values())
        std_likes = np.std(like_dist.values())
        adjusted_mean_likes = 1000 * float(mean_likes)/float(data["followers_count"])
        save_json(like_dist, os.path.join(save_dir, "like_dist.json"))
        create_dist_graph(like_dist, "like_dist")
        mean_retweets = np.mean(retweet_dist.values())
        std_retweets = np.std(retweet_dist.values())
        adjusted_mean_retweets = 1000 * float(mean_retweets)/float(data["followers_count"])
        save_json(retweet_dist, os.path.join(save_dir, "retweet_dist.json"))
        create_dist_graph(retweet_dist, "retweet_dist")
        filename = os.path.join(save_dir, "interarrivals.txt")
        with open(filename, 'w') as handle:
            std = np.std(data["interarrivals"].values())
            handle.write("Standard deviation: " + str(std) + "\n")
            for key in sorted(data["interarrivals"].keys()):
                outstring = str(key) + " | " + str(data["interarrivals"][key]) + "\n"
                handle.write(outstring.encode('utf-8'))

        filename = os.path.join(save_dir, "tweets.txt")
        data["tweet_texts"] = []
        with io.open(filename, 'w', encoding="utf-8") as handle:
            for t in tweet_texts:
                handle.write(t + u"\n")
                data["tweet_texts"].append(t)

        filename = os.path.join(save_dir, "tweet_interactions.csv")
        save_gephi_csv(data["interactions"], filename)

        data["heatmap"] = create_heatmap(tweet_timestamps)
        filename = os.path.join(save_dir, "heatmap.csv")
        save_heatmap(data["heatmap"], filename)

        filename = os.path.join(save_dir, "digest.txt")
        with io.open(filename, 'w', encoding='utf-8') as handle:
            handle.write(u"User name: " + data["name"] + u"\n")
            handle.write(u"Screen name: @" + data["screen_name"] + u"\n")
            handle.write(u"User id: " + unicode(data["user_id"]) + u"\n")
            handle.write(u"Tweets: " + unicode(data["statuses_count"]) + u"\n")
            handle.write(u"Likes: " + unicode(data["favourites_count"]) + u"\n")
            handle.write(u"Lists: " + unicode(data["listed_count"]) + u"\n")
            handle.write(u"Following: " + unicode(data["friends_count"]) + u"\n")
            handle.write(u"Followers: " + unicode(data["followers_count"]) + u"\n")
            handle.write(u"Created: " + unicode(data["account_created_date_readable"]) + u"\n")
            handle.write(u"Description: " + unicode(data["description"]) + u"\n")
            handle.write(u"Tweets per hour: " + unicode(data["tweets_per_hour"]) + u"\n")
            handle.write(u"Tweets per day: " + unicode(data["tweets_per_day"]) + u"\n")
            handle.write(u"Original tweets: " + unicode(data["original_tweet_count"]) + u"\n")
            handle.write(u"Retweets: " + unicode(data["retweet_count"]) + u"\n")
            handle.write(u"Tweets: " + unicode(data["tweet_count"]) + u"\n")
            handle.write(u"Replies: " + unicode(data["reply_count"]) + u"\n")
            handle.write(u"Total likes: " + unicode(total_likes) + u"\n")
            handle.write(u"Most liked: " + unicode(most_liked) + u"\n")
            handle.write(most_liked_tweet + u"\n")
            handle.write(u"Mean likes/tweet: " + unicode(mean_likes) + u"\n")
            handle.write(u"Std likes/tweet: " + unicode(std_likes) + u"\n")
            handle.write(u"Adjusted mean likes/tweet: " + unicode(adjusted_mean_likes) + u"\n")
            handle.write(u"Total retweets: " + unicode(total_retweets) + u"\n")
            handle.write(u"Most retweeted: " + unicode(most_retweeted) + u"\n")
            handle.write(most_retweeted_tweet + u"\n")
            handle.write(u"Mean retweets/tweet: " + unicode(mean_retweets) + u"\n")
            handle.write(u"Std retweets/tweet: " + unicode(std_retweets) + u"\n")
            handle.write(u"Adjusted mean retweets/tweet: " + unicode(adjusted_mean_retweets) + u"\n")
            data_string = output_data(10)
            handle.write(data_string)

        data_string = output_data()
        filename = os.path.join(save_dir, "full.txt")
        with io.open(filename, 'w', encoding='utf-8') as handle:
            handle.write(data_string)

        dump_chronology(save_dir)

        print
        print "Getting followers for " + target
        data["followers_ids"] = auth_api.followers_ids(target)
        data["followers_details"] = get_user_objects(data["followers_ids"])
        print
        print("Analyzing followers")
        bot_followers = set()
        analyze_account_creation_dates(data["followers_details"], save_dir, "followers")
        for user in data["followers_details"]:
            screen_name = user["screen_name"]
            if is_new_account_bot(user):
                bot_followers.add(screen_name)
        data["bot_follow_list"] = [x for x in sorted(bot_followers)]
        filename = os.path.join(save_dir, "bot_followers.json")
        save_json(data["bot_follow_list"], filename)
        data["follower_ranges"] = make_ranges(data["followers_details"])
        filename = os.path.join(save_dir, "follower_ranges.txt")
        with open(filename, "w") as f:
            f.write("Follower account age ranges\n")
            f.write("===========================\n\n")
            for key, val in sorted(data["follower_ranges"].items()):
                f.write(key + ":\t" + str(len(val)) + "\n")

        print "Getting friends for " + target
        data["friends_ids"] = auth_api.friends_ids(target)
        data["friends_details"] = get_user_objects(data["friends_ids"])
        print
        print("Analyzing friends")
        bot_friends = set()
        analyze_account_creation_dates(data["friends_details"], save_dir, "friends")
        for user in data["friends_details"]:
            screen_name = user["screen_name"]
            if is_new_account_bot(user):
                bot_friends.add(screen_name)
        data["bot_friend_list"] = [x for x in sorted(bot_friends)]
        filename = os.path.join(save_dir, "bot_friends.json")
        save_json(data["bot_friend_list"], filename)
        data["friends_ranges"] = make_ranges(data["friends_details"])
        filename = os.path.join(save_dir, "friends_ranges.txt")
        with open(filename, "w") as f:
            f.write("Friends account age ranges\n")
            f.write("==========================\n\n")
            for key, val in sorted(data["friends_ranges"].items()):
                f.write(key + ":\t" + str(len(val)) + "\n")

        account_interactions = {}
        account_interactions[data["screen_name"]] = Counter()
        for d in data["followers_details"]:
            if "screen_name" in d:
                account_interactions[data["screen_name"]][d["screen_name"]] += 1
        for d in data["friends_details"]:
            if "screen_name" in d:
                account_interactions[data["screen_name"]][d["screen_name"]] += 1
        filename = os.path.join(save_dir, "account_interactions.csv")
        save_gephi_csv(account_interactions, filename)

        filename = os.path.join(save_dir, "all_data.json")
        save_json(data, filename)

        if save_images == True:
            print("Fetching images")
            images_dir = os.path.join(save_dir, "images")
            if not os.path.exists(images_dir):
                os.makedirs(images_dir)
            dump_images(image_urls, images_dir)

