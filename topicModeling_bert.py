import json
from top2vec import Top2Vec
from topicModeling import preprocessing
from sentence_transformers import SentenceTransformer
import os
import umap
import hdbscan
import matplotlib.pyplot as plt
import pandas as pd

try:
    cleaned_tweets = []
    # with open('data/cleaned_adhd_text.txt', 'r') as f:
    #     for line in f:
    #         cleaned_tweets.append(line.strip())

    with open('data/all_adhd_user.json') as f:
        data = json.load(f)
    origin_tweet = [i['text'] for i in data if not i['referenced_tweets']]

    # tokens = preprocessing(origin_tweet)

    # for tweet in origin_tweet:
    #     cleaned_tweets.append(' '.join(preprocessing(tweet)))
    cleaned_tweets = origin_tweet

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print('sentence transformer...')
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    embeddings = model.encode(cleaned_tweets, show_progress_bar=True)

    print('umap...')
    umap_embeddings = umap.UMAP(n_neighbors=15,
                                n_components=5,
                                metric='cosine').fit_transform(embeddings)

    print('hsbscan...')
    cluster = hdbscan.HDBSCAN(min_cluster_size=15,
                              metric='euclidean',
                              cluster_selection_method='eom').fit(umap_embeddings)

    # Prepare data
    print('plotting...')
    umap_data = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
    result = pd.DataFrame(umap_data, columns=['x', 'y'])
    result['labels'] = cluster.labels_

    # Visualize clusters
    fig, ax = plt.subplots(figsize=(20, 10))
    outliers = result.loc[result.labels == -1, :]
    clustered = result.loc[result.labels != -1, :]
    plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=0.05)
    plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=0.05, cmap='hsv_r')
    plt.colorbar()
    plt.show()

    docs_df = pd.DataFrame(cleaned_tweets, columns=["Doc"])
    docs_df['Topic'] = cluster.labels_
    docs_df['Doc_ID'] = range(len(docs_df))
    docs_per_topic = docs_df.groupby(['Topic'], as_index = False).agg({'Doc': ' '.join})

    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer


    def c_tf_idf(documents, m, ngram_range=(1, 1)):
        # preprocessr should return ' '.join(tokens)
        count = CountVectorizer(ngram_range=ngram_range, preprocessor=preprocessing).fit(documents)
        t = count.transform(documents).toarray()
        w = t.sum(axis=1)
        tf = np.divide(t.T, w)
        sum_t = t.sum(axis=0)
        idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
        tf_idf = np.multiply(tf, idf)

        return tf_idf, count

    print('tf-idf...')
    tf_idf, count = c_tf_idf(docs_per_topic.Doc.values, m=len(cleaned_tweets))

    def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
        words = count.get_feature_names()
        labels = list(docs_per_topic.Topic)
        tf_idf_transposed = tf_idf.T
        indices = tf_idf_transposed.argsort()[:, -n :]
        top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
        return top_n_words

    def extract_topic_sizes(df):
        topic_sizes = (df.groupby(['Topic'])
                         .Doc
                         .count()
                         .reset_index()
                         .rename({"Topic": "Topic", "Doc": "Size"}, axis='columns')
                         .sort_values("Size", ascending=False))
        return topic_sizes

except Exception as e:
    print(e)

print('extracting...')
top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20)
topic_sizes = extract_topic_sizes(docs_df)

for i in range(20):
    idx = topic_sizes.at[i, 'Topic']
    if idx != -1:
        print(top_n_words[idx][:10])
