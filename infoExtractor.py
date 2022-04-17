import tweepy
import constant
import json
import time
from tqdm import tqdm

def get_tweet_info(userid, func, tweet_fields, max_results):
    """general method to extract tweets-related info"""
    tweet_response = func(id=userid, max_results=max_results, tweet_fields=tweet_fields)
    tweet_data = tweet_response.data
    if 'next_token' in tweet_response.meta.keys():
        tweet_next_token = tweet_response.meta['next_token']

        try:
            while True:
                tweet_response = func(id=userid, max_results=max_results,
                                      tweet_fields=tweet_fields, pagination_token=tweet_next_token)

                if 'next_token' in tweet_response.meta.keys():
                    tweet_next_token = tweet_response.meta['next_token']
                    tweet_data.extend(tweet_response.data)
                    time.sleep(0.5)
                else:
                    break

        except Exception as e:
            print(e)

    # print(tweet_data[0].referenced_tweets[0])
    # print(type(tweet_data[0].referenced_tweets[0]))

    tweet_list = [{'tweet_id': tw.id, 'text': tw.text, 'created_at': str(tw.created_at),
                   'public_metrics': tw.public_metrics, 'in_reply_to_user_id': tw.in_reply_to_user_id,
                   'referenced_tweets': {'id': tw.referenced_tweets[0].id, 'type':tw.referenced_tweets[0].type} if tw.referenced_tweets else []
                   } for tw in tweet_data]

    return tweet_list


if __name__ == "__main__":
    client = tweepy.Client(bearer_token=constant.academic_bearer_token,
                           consumer_key=constant.academic_api_key,
                           consumer_secret=constant.academic_api_key_secret,
                           access_token=constant.academic_access_token,
                           access_token_secret=constant.academic_access_token_secret,
                           wait_on_rate_limit=True)

    with open('data/filtered_random_user.txt', 'r') as f:
        data = [int(line.strip()) for line in f]

    expansion = ['author_id']
    tweet_fields = ['created_at', 'geo', 'public_metrics', 'context_annotations', 'referenced_tweets', 'in_reply_to_user_id']
    follower_list = []
    following_list = []
    tweets_list = []

    # user_ids = [tw['author_id'] for tw in data]
    user_ids = data

    for i in tqdm(range(3300)):
        try:
            # get the user tweets
            tweets_list = get_tweet_info(user_ids[i], client.get_users_tweets, tweet_fields, 100)
            # print(tweets_list)
            #
            # # get users' liked tweets
            # liked_tweets_list = get_tweet_info(userid, client.get_liked_tweets, tweet_fields, 5)
            # print(liked_tweets_list)
            #
            # # get users' followers
            # follower_response = client.get_users_followers(id=userid, max_results=10)
            # follower_next_token = follower_response['next_token']
            # for user in follower_response.data:
            #     print(user.id)
            #
            # # get users' following
            # following_response = client.get_users_following(id=userid, max_results=10)
            # following_next_token = following_response['next_token']
            # print(follower_response)

            with open('data/random_users_tweets/'+str(user_ids[i])+'.json', 'w') as f:
                json.dump(tweets_list, f, indent=4)

            # print('User', str(user_ids[i]), 'collected.')

        except Exception as e:
            print('User', str(user_ids[i]))
            print(e)
            with open('data/random_users_error.txt', 'a', encoding='utf-8') as f:
                f.write(str(user_ids[i])+'\n')

