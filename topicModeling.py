import itertools

import numpy as np
import pandas as pd
import json
import os
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize, TweetTokenizer
from nltk.stem import WordNetLemmatizer
import string
import re
import nltk
import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from constant import emoji_regex
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from slang import slangs
from sklearn.cluster import KMeans
from tqdm import tqdm



# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')


def count_emoji(data):
    all_text = ' '.join([tw['text'] for tw in data]).lower()
    counter_dict = Counter()
    emoji_array = emoji_regex.findall(all_text)
    counter_dict.update(emoji_array)
    return pd.DataFrame.from_dict(counter_dict, orient='index')


def preprocessing(data):
    # print(data)
    # all_text = ' '.join(data).lower()  # input all tweets
    all_text = data.lower()  # input one single tweet

    # translate abbreviated slang
    # print('translating slang...')
    # with open('slang.txt', 'r') as f:
    #     slang = [{'ab': line.split('=')[0].strip().lower(), 'text': line.split('=')[1].strip().lower()} for line in f]

    # remove urls
    print('removing urls...')
    all_text = re.sub(r"http\S+", "", all_text)
    # texts = [re.sub(r"http\S+", "", t) for t in texts]

    # remove number
    print('removing numbers...')
    all_text = re.sub(r'\d+', '', all_text)
    # texts = [re.sub(r'\d+', '', t) for t in texts]

    # remove retweet sign
    all_text = all_text.replace('rt', '')
    # texts = [t.replace('rt', '') for t in texts]

    # remove emoji
    all_text = re.sub(r'[^\x00-\x7F]+', '', all_text)
    # texts = [re.sub(r'[^\x00-\x7F]+', '', t) for t in texts]

    # remove punctuation
    """To do: keep part of punctuation (... & !!.) and more!"""
    print('removing punctuation...')

    def remove_punctuation(words):
        table = str.maketrans('', '', string.punctuation)
        return words.translate(table)

    all_text = remove_punctuation(all_text)
    # texts = [remove_punctuation(t) for t in texts]

    # stopwords
    stop_words = nltk.corpus.stopwords.words('english')
    stop_words += stopwords.stopwords
    stop_words += ["’", "…", "“", "”", "️", ]

    # Tokenizing
    print('tokenizing...')
    tt = TweetTokenizer()
    tokens = tt.tokenize(all_text)
    # tokens = [tt.tokenize(t) for t in texts]
    stopwords_removed_tokens = [token for token in tokens if token not in stop_words and token != '']
    # stopwords_removed_tokens = [t for token in tokens for t in token if t not in stop_words and t != '']

    # Lemmatizing
    print('lemmatizing...')
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in stopwords_removed_tokens]
    # lemmatized_tokens = [lemmatizer.lemmatize(w) for word in stopwords_removed_tokens for w in word]

    return lemmatized_tokens  # for top2vec
    # return " ".join(lemmatized_tokens)  # for bert


def generate_wordCloud(tokens, k=30):
    # flat_tokens = list(itertools.chain(*tokens))
    print('generating wordCloud')
    # word_freq = FreqDist(flat_tokens)
    word_freq = FreqDist(tokens)
    print(word_freq.most_common((100)))
    most_common_count = [x[1] for x in word_freq.most_common(k)]
    most_common_word = [x[0] for x in word_freq.most_common(k)]

    # create dictionary mapping of word count
    top_dictionary = dict(zip(most_common_word, most_common_count))

    # Create Word Cloud of top 30 words
    wordcloud = WordCloud(colormap='Accent', background_color='black').generate_from_frequencies(top_dictionary)

    # plot with matplotlib
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig('top_30_cloud.png')

    plt.show()


def topicModeling(tokens):
    print('Topic Modeling...')
    # tf-idf vectorization
    texts = [" ".join(token) for token in tokens]
    df = pd.DataFrame(texts)
    vect = TfidfVectorizer(sublinear_tf=True, min_df=5, max_df=0.95)
    vect_text = vect.fit_transform(texts)

    # # kmeans
    # kmeans = KMeans(n_clusters=5, random_state=42)
    # kmeans.fit(vect_text)
    # clusters = kmeans.labels_
    #
    # def get_top_keywords(n_terms):
    #     """This function returns the keywords for each centroid of the KMeans"""
    #     df = pd.DataFrame(vect_text.todense()).groupby(clusters).mean()  # groups the TF-IDF vector by cluster
    #     terms = vect.get_feature_names_out()  # access tf-idf terms
    #     for i, r in df.iterrows():
    #         print('\nCluster {}'.format(i))
    #         print(','.join([terms[t] for t in np.argsort(r)[
    #                                           -n_terms:]]))  # for each row of the dataframe, find the n terms that have the highest tf idf score
    #
    # get_top_keywords(10)

    # lda
    lda_model = LatentDirichletAllocation(n_components=10, learning_method='online', random_state=42, max_iter=1)
    lda_top = lda_model.fit_transform(vect_text)

    print("Document 0: ")
    for i, topic in enumerate(lda_top[0]):
        print("Topic ", i, ": ", topic * 100, "%")

    vocab = vect.get_feature_names()
    for i, comp in enumerate(lda_model.components_):
        vocab_comp = zip(vocab, comp)
        sorted_words = sorted(vocab_comp, key=lambda x: x[1], reverse=True)[:10]
        print("Topic " + str(i) + ": ")
        for t in sorted_words:
            print(t[0], end=" ")
        print("\n")


def main():
    # read ADHD gorup users
    print('Analyzing ADHD group...')
    with open('data/all_random_user.json') as f:
        data = json.load(f)
    origin_tweet = [i['text'] for i in data if not i['referenced_tweets']]

    # tokens = preprocessing(origin_tweet)

    tokens = []
    for tweet in tqdm(origin_tweet):
        tokens.append(preprocessing(tweet))
    topicModeling(tokens)

    # generate_wordCloud(tokens, 60)

    # emoji_counter = count_emoji(data)
    # # print(emoji_counter)
    # emoji_counter.to_csv('data/emoji_count.csv')
    #
    # # read random gorup users
    # print('Analyzing random group...')
    # path = 'data/random_users_tweets'
    # data = readData(path)
    #
    # tokens = preprocessing(data)
    #
    # generate_wordCloud(tokens, 60)
    #
    # emoji_counter = count_emoji(data)
    # # print(emoji_counter)
    # emoji_counter.to_csv('data/random_user_emoji_count.csv')

    # print('Analyzing non-ADHD group...')
    # with open('data/all_random_user.json') as f:
    #     data = json.load(f)
    # origin_tweet = [i['text'] for i in data if not i['referenced_tweets']]
    #
    # # tokens = preprocessing(origin_tweet)
    #
    # tokens = []
    # for tweet in origin_tweet:
    #     tokens.extend(preprocessing(tweet))
    # # topicModeling(tokens)
    #
    # generate_wordCloud(tokens, 60)

    # emoji_counter = count_emoji(data)
    # # print(emoji_counter)
    # emoji_counter.to_csv('data/emoji_count.csv')
    #
    # # read random gorup users
    # print('Analyzing random group...')
    # path = 'data/random_users_tweets'
    # data = readData(path)
    #
    # tokens = preprocessing(data)
    #
    # generate_wordCloud(tokens, 60)
    #
    # emoji_counter = count_emoji(data)
    # # print(emoji_counter)
    # emoji_counter.to_csv('data/random_user_emoji_count.csv')


if __name__ == "__main__":
    main()
