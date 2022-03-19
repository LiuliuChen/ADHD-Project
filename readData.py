import os
import json

if __name__ == "__main__":
    data = []
    path = 'data/users_tweets'
    files = os.listdir(path)
    # read all data under users_tweets
    for file in files:
        # file is not None
        if os.path.getsize(path + '/' + file) != 0:
            with open(path + '/' + file, 'r') as f:
                data.extend(json.load(f))
