import tweepy
import constant
import json
import time


if __name__ == "__main__":
    client = tweepy.Client(bearer_token=constant.academic_bearer_token,
                           consumer_key=constant.academic_api_key,
                           consumer_secret=constant.academic_api_key_secret,
                           access_token=constant.academic_access_token,
                           access_token_secret=constant.academic_access_token_secret,
                           wait_on_rate_limit=True)

    adhd_data_dict = []
    user_id_list = []
    query = '((I OR my) (got OR have OR am OR (just got) OR (was diagnosed with)) ADHD) -is:retweet -is:reply -is:quote lang:en'
    fields = ["id", "text", "author_id", "created_at", "public_metrics"]
    expansions = 'author_id'
    adhd_tweets = client.search_all_tweets(query=query, max_results=500, tweet_fields=fields, expansions=expansions)
    # print(adhd_tweets.includes['users'][0].id)
    adhd_data_dict = adhd_tweets.data
    user_data_list = adhd_tweets.includes['users']
    next_token = adhd_tweets.meta['next_token']

    # print(adhd_tweets.data[0]['created_at'])
    # print('next token: ', next_token)

    count=1

    try:
        while True:
            adhd_tweets = client.search_all_tweets(query=query, max_results=500,
                                                   next_token=next_token, expansions=expansions)
            adhd_data_dict.extend(adhd_tweets.data)
            user_data_list.extend(adhd_tweets.includes['users'])
            # get next page of users
            next_token = adhd_tweets.meta['next_token']
            time.sleep(2)

            if next_token is None:
                break

            if len(adhd_data_dict) > 30000:
                print('Next Token: ', next_token)
                break

            print('Collected: ', count*500)
            count += 1

        adhd_data = [{"author_id": tw[1].id, "tweet_id": tw[0].id, "created_at": str(tw[0].created_at),
                      "text": tw[0].text, "public_metrics": tw[0].public_metrics} for tw in zip(adhd_data_dict, user_data_list)]
        with open('data/adhd_content_3.json', 'w') as f:
            json.dump(adhd_data, f, indent=4)

    except Exception as e:
        print(e)
        adhd_data = [{"author_id":tw.author_id, "tweet_id": tw.id, "created_at": str(tw.created_at),
                      "text": tw.text} for tw in adhd_data_dict]
        with open('data/adhd_content_3.json', 'w') as f:
            json.dump(adhd_data, f, indent=4)


