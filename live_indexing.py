from __future__ import absolute_import, division, print_function
import nltk
from collections import defaultdict
from nltk.stem.snowball import EnglishStemmer  # Assuming we're working with English
import pymongo
import faiss
import pandas as pd
import dask.dataframe as dd
from dask import delayed, compute
import numpy as np
from pympler import tracker
from pympler.asizeof import asizeof
# nltk.download('stopwords')
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from cache_to_disk import cache_to_disk
from cache_to_disk import delete_disk_caches_for_function
from sentence_transformers import SentenceTransformer
from joblib import dump, load
import collections
import logging
import math

import torch
from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForQuestionAnswering, BertTokenizer)
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

from utils import (get_answer, input_to_squad_example,
                   squad_examples_to_features, to_list)

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])

from os import path
import os 

from cloud_storage import test_file_exists, download_blob, upload_blob, pull_indices

# delete_disk_caches_for_function('get_data')
collection = pymongo.MongoClient('mongodb+srv://ir2:HUADLhhOLoCQ02VS@cluster1.xo9vl.mongodb.net/document?retryWrites=true&w=majority').document.document
mongo_query = lambda i,batch_size: [
    {
        '$project': {
            'content': {
                '$reduce': {
                    'input': '$content.text', 
                    'initialValue': '', 
                    'in': {
                        '$concat': [
                            '$$value', ' ', '$$this'
                        ]
                    }
                }
            }, 
            'title': 1, 
            'description': 1
        }
    }, {
        '$project': {
            'text': {
                '$concat': [
                    '$title', ' ', '$description', ' ', '$content'
                ]
            }
        }
    },
    { "$limit":  i+batch_size },
    { "$skip": i }
]

def build_faiss(tfidf_model, bert_model):
    tr = tracker.SummaryTracker()
    print(f"Building indices ...")
    c = collection.find().count()
    current_n = tfidf_model.n
    # c = 1000
    batch_size = 500
    encoder = None
    bert_index =  None
    tfidf_index = None
    # if hasattr(model, 'encode'):
    #     encoder =  lambda x: model.encode(x).astype("float32")
    # else:
    #     encoder = lambda x:model.transform(x).toarray().astype("float32")
    i = 0
    ids = []
    while i < c:
        print(i)
        docs = []
        for x in collection.aggregate(mongo_query(i,batch_size)) :
            # docs.append(x.get("title","") + " " + x.get('description',"")+ " " + " ".join(filter(None,x.get('content',{}).get('text',[]))))
            docs.append(x['text'] or "")
            ids.append(x['_id'])
        print("Downloaded batch",i)
        tfidf_embeddings = tfidf_model.transform(docs).toarray().astype("float32")
        print("Computed tfidf embeddings")
        bert_embeddings = bert_model.encode([doc[:100] for doc  in docs]).astype("float32")
        print("Computed bert embeddings")
        if i == 0:
            bert_index = faiss.IndexFlatIP(bert_embeddings.shape[1])
            tfidf_index = faiss.IndexFlatIP(tfidf_embeddings.shape[1])
            
        # print(bert_embeddings.shape[1])
        # print(tfidf_embeddings.shape[1])
        faiss.normalize_L2(bert_embeddings)
        faiss.normalize_L2(tfidf_embeddings)
        print(tr.print_diff())

        # Step 3: Pass the index to IndexIDMap
        # index = faiss.IndexIDMap(index)
        # Step 4: Add vectors and their IDs
        # print("range",len(np.arange(i,i+len(embeddings))))
        # print("embeds",len(embeddings))
        # idmap.add_with_ids(embeddings,np.arange(i,i+len(embeddings)))

        bert_index.add(bert_embeddings)
        tfidf_index.add(tfidf_embeddings)
        i += len(tfidf_embeddings)
    faiss.write_index(bert_index,f"models/bert.index")
    faiss.write_index(tfidf_index,f"models/tfidf.index")
    dump(ids,'models/ids.joblib')
    print(f"Completed indices.")
    # upload_indices_and_vectors()
    return [tfidf_index, bert_index]

def load_faiss(tfidf_model, bert_model):
    if path.exists(f"models/bert.index") and path.exists(f"models/tfidf.index"):
        return [faiss.read_index(f"models/tfidf.index"),faiss.read_index(f"models/bert.index")]
    else:
        return build_faiss(tfidf_model, bert_model)
