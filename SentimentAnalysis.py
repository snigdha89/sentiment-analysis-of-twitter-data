import twint
import nest_asyncio
import pandas as pd 
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from torch import nn
from nrclex import NRCLex
from transformers import pipeline
nest_asyncio.apply()

# Configure
c = twint.Config()
c.Search = "\"Booster shot\"" "Covid-19"
c.Store_csv = True
c.Output = "./Assignment4_output.csv"
c.Lang = "en"
c.Limit = 20
twint.run.Search(c)

data = pd.read_csv("./Assignment4_output.csv")
final = data['tweet'].str.lower()
tweet_list = final.tolist()

for i in range(len(tweet_list)):
    emotion = NRCLex(tweet_list[i])
    print('\n\n', tweet_list[i], ': ', emotion.top_emotions)

classifier = pipeline('sentiment-analysis')
classifier(tweet_list)
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
pt_batch = [tokenizer([tweet], padding = True, truncation = True, return_tensors = 'pt') for tweet in tweet_list]
pt_outputs = [pt_model(**x) for x in pt_batch]
pt_predictions = [nn.functional.softmax(y.logits, dim = -1) for y in pt_outputs]
pt_predictions




