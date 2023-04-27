import json
from top2vec import Top2Vec
from topicModeling import preprocessing

try:
    cleaned_tweets = []
    with open('data/all_random_user.json') as f:
        data = json.load(f)
    origin_tweet = [i['text'] for i in data if not i['referenced_tweets']]

    cleaned_tweets = origin_tweet

    # tokenizer should return a list of tokens
    model = Top2Vec(documents=cleaned_tweets, speed='learn', workers=30, tokenizer=preprocessing)

    topic_words, word_scores, topic_nums = model.get_topics()

    print(topic_words)
    print(word_scores)
    print(topic_nums)

    with open('data/non_adhd_top2vec_res.txt', 'a', encoding='utf-8') as f:
        for i in topic_words:
            f.write(' '.join(i))
            f.write('\n')

    model.save('../autodl-tmp/adhd/non_adhd_top2vec_trained')

    import os
    os.system("shutdown")

except Exception as e:
    print(e)
