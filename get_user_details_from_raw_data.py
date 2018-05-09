from process_tweet_object import *
import json
import os
import sys
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

if __name__ == '__main__':
    if (len(sys.argv) < 2):
        print("Please supply input file path.")
        sys.exit(0)

    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print("Could not load " + input_file)
        sys.exit(0)
    save_dir = os.path.dirname(input_file)
    print("Save dir: " + save_dir)

    print("Loading data from " + input_file)
    user_details = {}
    user_count = 0
    with io.open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            status = json.loads(line)
            if "user" in status and status["user"] is not None:
                user = status["user"]
                id_str = user_get_id_str(user)
                details = get_user_details_dict(user)
                if id_str not in user_details:
                    user_details[id_str] = details
                    user_count += 1
                    print("Found " + str(user_count) + " unique users.")
    filename = os.path.join(save_dir, "user_details.json")
    print("Saving " + filename)
    save_json(user_details, filename)
