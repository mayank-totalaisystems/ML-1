# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 15:03:53 2021

@author: Admin
"""

from flask import Flask
from flask import request
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer, util
import numpy as np
import spacy
import pickle
import json
app = Flask(__name__)

@app.route('/classify', methods=['GET'])
def welcome():
    sentence = request.args.get('sentence')
    sentence_embedding = model.encode(sentence, convert_to_tensor=True)
# top_k results to return
    top_k=1
    # compute similarity scores of the sentence with the corpus
    cos_scores = util.pytorch_cos_sim(sentence_embedding, corpus_embeddings)[0]
    # Sort the results in decreasing order and get the first top_k
    top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]
    print("Sentence:", sentence, "\n")
    print("Top", top_k, "most similar sentences in corpus:")
    for idx in top_results[0:top_k]:
        print(corpus[idx], "(Score: %.4f)" % (cos_scores[idx]))
        index=corpus.index(corpus[idx])
        
    Tag=label[index]
    dict={}
    doc = nlp(sentence)
    for entity in doc.ents:
        dict[entity.label_] = entity.text

    result={'lable':Tag,'entities':dict}
    return(json.dumps(result))


if __name__ == '__main__':
    file=pd.read_csv("replies.csv")
    model = SentenceTransformer('stsb-roberta-large')
    corpus = list(file['ans1'])
    label=list(file['Tag'])
    nlp = spacy.load('en_core_web_sm')
    # encode corpus to get corpus embeddings
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
    app.run()