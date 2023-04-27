import json
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer


# Create class for data preparation
class SimpleDataset:
    def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts

    def __len__(self):
        return len(self.tokenized_texts["input_ids"])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tokenized_texts.items()}


if __name__ == '__main__':

    # Load tokenizer and model, create trainer
    model_name = "joeddav/distilbert-base-uncased-go-emotions-student"
    # model_name = "explosion/en_textcat_goemotions"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    trainer = Trainer(model=model)


    # import data
    idx = 0
    with open('../autodl-tmp/adhd/raw_data/all_non_adhd_users.json') as f:
        data = json.load(f)
    # tweets = [i['text'] for i in data if not i['referenced_tweets']]
    # tweets_id = [i['tweet_id'] for i in data if not i['referenced_tweets']]
    # author_id = [i['author_id'] for i in data if not i['referenced_tweets']]
    tweets = [i['text'] for i in data if i['referenced_tweets'] == "[]"]
    tweets_id = [i['tweet_id'] for i in data if i['referenced_tweets'] == "[]"]
    author_id = [i['author_id'] for i in data if i['referenced_tweets'] == "[]"]
    print(len(tweets))
    part_tweets = tweets[idx: idx + 50000]
    part_tweets_id = tweets_id[idx: idx + 50000]
    part_author_id = author_id[idx: idx + 50000]

    while True:

        # work in progress
        # container
        admiration = []
        amusement = []
        anger = []
        annoyance = []
        approval = []
        caring = []
        confusion = []
        curiosity = []
        desire = []
        disappointment = []
        disapproval = []
        disgust = []
        embarrassment = []
        excitement = []
        fear = []
        gratitude= []
        grief = []
        joy = []
        love = []
        nervousness = []
        optimism = []
        pride = []
        realization = []
        relief = []
        remorse = []
        sadness = []
        surprise = []
        neutral = []

        # Tokenize texts and create prediction data set
        # pred_texts = ['I love you!']
        print('tokenizing...')
        tokenized_texts = tokenizer(part_tweets, truncation=True, padding=True)
        pred_dataset = SimpleDataset(tokenized_texts)

        # Run predictions
        print('predicting...')
        predictions = trainer.predict(pred_dataset)


        # Transform predictions to labels
        preds = predictions.predictions.argmax(-1)
        labels = pd.Series(preds).map(model.config.id2label)
        scores = (np.exp(predictions[0])/np.exp(predictions[0]).sum(-1,keepdims=True)).max(1)
        # print(labels, scores)

        # scores raw
        temp = (np.exp(predictions[0])/np.exp(predictions[0]).sum(-1,keepdims=True))
        # print(temp)

        for j in range(len(part_tweets)):
            admiration.append(temp[j][0])
            amusement.append(temp[j][1])
            anger.append(temp[j][2])
            annoyance.append(temp[j][3])
            approval.append(temp[j][4])
            caring.append(temp[j][5])
            confusion.append(temp[j][6])
            curiosity.append(temp[j][7])
            desire.append(temp[j][8])
            disappointment.append(temp[j][9])
            disapproval.append(temp[j][10])
            disgust.append(temp[j][11])
            embarrassment.append(temp[j][12])
            excitement.append(temp[j][13])
            fear.append(temp[j][14])
            gratitude.append(temp[j][15])
            grief.append(temp[j][16])
            joy.append(temp[j][17])
            love.append(temp[j][18])
            nervousness.append(temp[j][19])
            optimism.append(temp[j][20])
            pride.append(temp[j][21])
            realization.append(temp[j][22])
            relief.append(temp[j][23])
            remorse.append(temp[j][24])
            sadness.append(temp[j][25])
            surprise.append(temp[j][26])
            neutral.append(temp[j][27])



        # Create DataFrame with texts, predictions, labels, and scores
        df = pd.DataFrame(list(zip(part_author_id, part_tweets_id, part_tweets, preds, labels, scores, admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral)), columns=['author_id','tweet_id','tweet','pred','label','score', 'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'])
        df.to_csv('../autodl-tmp/adhd/sentiment/non_adhd_sentimental_analysis(goEmotion)' + str(idx) + '.csv', index=False)

        if idx >= len(tweets):
            break

        if idx + 50000 >= len(tweets):
            part_tweets = tweets[idx:]
            part_tweets_id = tweets_id[idx:]
            part_author_id = author_id[idx:]
            idx += 50000
        else:
            idx += 50000
            part_tweets = tweets[idx: idx + 50000]
            part_tweets_id = tweets_id[idx: idx + 50000]
            part_author_id = author_id[idx: idx + 50000]