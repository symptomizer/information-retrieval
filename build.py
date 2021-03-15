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
# delete_disk_caches_for_function('get_data')
collection = pymongo.MongoClient('mongodb+srv://ir2:HUADLhhOLoCQ02VS@cluster1.xo9vl.mongodb.net/document?retryWrites=true&w=majority').document.document
all_docs = lambda: collection.find({}, projection={'title': True, 'description': True, "content.text": True})
doc_generator = lambda q=all_docs: ((x['title'] or "") + " " + x.get('description',"")+ " " + " ".join(filter(None,x['content']['text'] or [])) for x in collection.find({}, projection={'title': True, 'description': True, "content.text": True}))
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



def iterate_by_chunks(collection, chunksize=2, start_from=0, query={}, projection={ '_id': False }):
   chunks = range(start_from, collection.find(query,projection).count(), int(chunksize))
   num_chunks = 3 #len(chunks)
   for i in range(1,num_chunks+1):
        print("Loaded chunk ",i)
        if i < num_chunks:
            yield collection.find(query,projection)[chunks[i-1]:chunks[i]]
        else:
            yield collection.find(query,projection)[chunks[i-1]:chunks.stop]

def build_tfidf_model(num_docs=5):
    print("Building TF-IDF model...")
    model = TfidfVectorizer(ngram_range=(3,5), analyzer="char")
    model.fit((x.get("title","") + " " + x.get('description',"")+ " " + " ".join(filter(None,x.get('content',{}).get('text',[]))) for x in collection.find({}, projection={'title': True, 'description': True, "content.text": True}).limit(num_docs)))
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

def build_faiss(model, name):
    tr = tracker.SummaryTracker()
    print(f"Building {name} index ...")
    c = 2000 #collection.find({}, projection={'title': True, 'description': True, "content.text": True}).count()
    encoder = None
    index =  None
    # idmap = None
    if hasattr(model, 'encode'):
        encoder =  lambda x: model.encode(x).astype("float32")
    else:
        encoder = lambda x:model.transform(x).toarray().astype("float32")
    i = 0
    ids = []
    while i < c:
        print(i)
        docs = []
        for x in collection.find({}, projection={'_id': True, 'title': True, 'description': True, "content.text": True}).skip(i).limit(500):
            docs.append(x.get("title","") + " " + x.get('description',"")+ " " + " ".join(filter(None,x.get('content',{}).get('text',[]))))
            ids.append(x['_id'])
        print('docs',len(docs))
        embeddings = encoder(docs)
        if i == 0:
            index = faiss.IndexFlatIP(embeddings.shape[1])
            # idmap = faiss.IndexIDMap(index)
        
    
    # embeddings = np.array([embedding for embedding in embeddings]).astype("float32")

    # Step 2: Instantiate the index
        faiss.normalize_L2(embeddings)
        print(tr.print_diff())

    # Step 3: Pass the index to IndexIDMap
    # index = faiss.IndexIDMap(index)
    # Step 4: Add vectors and their IDs
        print("range",len(np.arange(i,i+len(embeddings))))
        print("embeds",len(embeddings))
        # idmap.add_with_ids(embeddings,np.arange(i,i+len(embeddings)))
        index.add(embeddings)
        i += len(embeddings)
    faiss.write_index(index,f"models/{name}.index")
    dump(ids,'models/ids.joblib')
    print(f"Completed {name} index.")
    return index
    

def load_faiss(model, name):
    if path.exists(f"models/{name}.index"):
        return faiss.read_index(f"models/{name}.index")
    else:
        return build_faiss(model, name)

        

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
