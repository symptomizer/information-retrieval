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
from preprocessing import preprocess_string, read_stop_words

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
# all_docs = lambda: collection.find({}, projection={'title': True, 'description': True, "content.text": True})
# doc_generator = lambda q=all_docs: ((x['title'] or "") + " " + x.get('description',"")+ " " + " ".join(filter(None,x['content']['text'] or [])) for x in collection.find({}, projection={'title': True, 'description': True, "content.text": True}))
# @cache_to_disk(2)
# def get_data():
#     print("Loading documents from databases ...")
#     # get the database
#     # meta = pd.DataFrame(list(collection.find({},{ '_id': False }).limit(1)))
#     # queries = iterate_by_chunks(collection,chunksize=2)
 
#     # for q in queries:
#     #     print(list(q))
#     #     # l = apply_q(q)
#     #     # print(l)
#     #     # df = dd.from_pandas(l,chunksize=2)
#     #     print("dione df")
#     for coll 
#         # dd.to_parquet(df,"models/documents")
    
#     # dd.to_parquet(df,"models/documents",overwrite=True)
    
#     # return pd.DataFrame(list(database.get_collection("document").find())).fillna('')



# def iterate_by_chunks(collection, chunksize=2, start_from=0, query={}, projection={ '_id': False }):
#    chunks = range(start_from, collection.find(query,projection).count(), int(chunksize))
#    num_chunks = 3 #len(chunks)
#    for i in range(1,num_chunks+1):
#         print("Loaded chunk ",i)
#         if i < num_chunks:
#             yield collection.find(query,projection)[chunks[i-1]:chunks[i]]
#         else:
#             yield collection.find(query,projection)[chunks[i-1]:chunks.stop]


def upload_indices_and_vectors():
    # Checks if PULL_INDS environment variable is present, and calls pull function
    if os.environ.get('PUSH_INDS') != None:
        upload_blob("symptomizer_indices_bucket-1", "models/bert.index", "bert.index")
        upload_blob("symptomizer_indices_bucket-1", "models/tfidf.index", "tfidf.index")
        upload_blob("symptomizer_indices_bucket-1", "models/ids.joblib", "ids.joblib")
        upload_blob("symptomizer_indices_bucket-1", "models/tfidf_model.joblib", "tfidf_model.joblib")
    else:
        print("No PUSH_INDS env found. Not pushing new index.")


def pull_and_preprocess_from_mongo(start_index, num_docs):
    
    docs = collection.aggregate(mongo_query(start_index, num_docs))
    doc_list = []
    id_list = []
    for doc in docs:
        clean_text = preprocess_string(doc['text'] or "", stopping = True, stemming = True, lowercasing = True)
        doc_list.append(clean_text)
        id_list.append(doc['_id'])
            
    return list(zip(doc_list, id_list))


def build_tfidf_model(num_docs=2000, max_features=3000):
    print("Building TF-IDF model...")
    model = TfidfVectorizer(ngram_range=(3,5), max_features=max_features, analyzer="char")
    data = pull_and_preprocess_from_mongo(0, num_docs)
    docs = [text for (text, ind) in data]
    model.fit(docs)
    dump(model, 'models/tfidf_model.joblib') 
    print("Saved TF-IDF model to models/tfidf_model.joblib.")
    return model

def load_tfidf_model():
    if path.exists(f"models/tfidf_model.joblib"):
        print("Loading TF-IDF model...")
        return load("models/tfidf_model.joblib")
    else:
        return build_tfidf_model()
    
def load_bert_model():
    print("Building BERT model...")
    model = SentenceTransformer('msmarco-distilbert-base-v2')
    print("Completed BERT model.")
    return model

    
def build_faiss(tfidf_model, bert_model):
    tr = tracker.SummaryTracker()
    print(f"Building indices ...")
    c = collection.find().count()
    # c = 5000
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
        for text, ind in pull_and_preprocess_from_mongo(i,batch_size):
            # docs.append(x.get("title","") + " " + x.get('description',"")+ " " + " ".join(filter(None,x.get('content',{}).get('text',[]))))
            docs.append(text)
            ids.append(ind)
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
    upload_indices_and_vectors()
    return [tfidf_index, bert_index]

def load_faiss(tfidf_model, bert_model):
    if path.exists(f"models/bert.index") and path.exists(f"models/tfidf.index"):
        return [faiss.read_index(f"models/tfidf.index"),faiss.read_index(f"models/bert.index")]
    else:
        return build_faiss(tfidf_model, bert_model)

        

class QA:

    def __init__(self,model_path: str):
        self.max_seq_length = 384
        self.doc_stride = 128
        self.do_lower_case = True
        self.max_query_length = 64
        self.n_best_size = 20
        self.max_answer_length = 30
        self.model, self.tokenizer = self.load_model(model_path)
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.model.to(self.device)
        self.model.eval()


    def load_model(self,model_path: str,do_lower_case=False):
        config = BertConfig.from_pretrained(model_path + "/bert_config.json")
        tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=do_lower_case)
        model = BertForQuestionAnswering.from_pretrained(model_path, from_tf=False, config=config)
        return model, tokenizer
    
    def predict(self,passage :str,question :str):
        example = input_to_squad_example(passage,question)
        features = squad_examples_to_features(example,self.tokenizer,self.max_seq_length,self.doc_stride,self.max_query_length)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_example_index)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=1)
        all_results = []
        for batch in eval_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2]  
                        }
                example_indices = batch[3]
                outputs = self.model(**inputs)

            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                result = RawResult(unique_id    = unique_id,
                                    start_logits = to_list(outputs[0][i]),
                                    end_logits   = to_list(outputs[1][i]))
                all_results.append(result)
        answer = get_answer(example,features,all_results,self.n_best_size,self.max_answer_length,self.do_lower_case)
        return answer

 
class Index:
    """ Inverted index datastructure """
 
    def __init__(self, tokenizer, stemmer=None, stopwords=None):
        """
        tokenizer   -- NLTK compatible tokenizer function
        stemmer     -- NLTK compatible stemmer 
        stopwords   -- list of ignored words
        """
        self.tokenizer = tokenizer
        self.stemmer = stemmer
        self.index = defaultdict(list)
        self.documents = {}
        self.__unique_id = 0
        if not stopwords:
            self.stopwords = set()
        else:
            self.stopwords = set(stopwords)
 
    def lookup(self, word):
        """
        Lookup a word in the index
        """
        word = word.lower()
        if self.stemmer:
            word = self.stemmer.stem(word)
 
        return [self.documents.get(id, None) for id in self.index.get(word)]
 
    def add(self, document):
        """
        Add a document string to the index
        """
        for token in [t.lower() for t in nltk.word_tokenize(document)]:
            if token in self.stopwords:
                continue
 
            if self.stemmer:
                token = self.stemmer.stem(token)
 
            if self.__unique_id not in self.index[token]:
                self.index[token].append(self.__unique_id)
 
        self.documents[self.__unique_id] = document
        self.__unique_id += 1           
 
 
index = Index(nltk.word_tokenize, 
              EnglishStemmer(), 
              nltk.corpus.stopwords.words('english'))
