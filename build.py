from __future__ import absolute_import, division, print_function
import nltk
from collections import defaultdict
from nltk.stem.snowball import EnglishStemmer  # Assuming we're working with English
import pymongo
import faiss
import pandas as pd
import numpy as np
nltk.download('stopwords')
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from cache_to_disk import cache_to_disk
from cache_to_disk import delete_disk_caches_for_function
from sentence_transformers import SentenceTransformer

import collections
import logging
import math

import numpy as np
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


@cache_to_disk(2)
def get_data():
    print("Loading documents from database ...")
    client = pymongo.MongoClient('mongodb+srv://ir:UE5Ki3Cr1eyWVeNl@cluster1.xo9vl.mongodb.net/myFirstDatabase?retryWrites=true&w=majority')
    # get the database
    database = client['document']
    return pd.DataFrame(list(database.get_collection("document").find())).fillna('')

def build_tfidf_model(documents):
    print("Building TF-IDF model...")
    model = TfidfVectorizer(ngram_range=(3,5), analyzer="char")
    model.fit_transform(documents.values.astype('U'))
    print("Completed TF-IDF model.")
    return model

def build_bert_model(documents=None):
    print("Building BERT model...")
    model = SentenceTransformer('msmarco-distilbert-base-v2')
    print("Completed BERT model.")
    return model

def build_faiss(model, documents, name):
    print(f"Building {name} index ...")
    if hasattr(model, 'encode'):
        embeddings = model.encode(documents).astype("float32")
    else:
        embeddings = model.transform(documents).toarray().astype("float32")

    # embeddings = np.array([embedding for embedding in embeddings]).astype("float32")

    # Step 2: Instantiate the index
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])

    # Step 3: Pass the index to IndexIDMap
    # index = faiss.IndexIDMap(index)
    # Step 4: Add vectors and their IDs
    index.add(embeddings)
    faiss.write_index(index,f"models/{name}.index")
    print(f"Completed {name} index.")
    return index
    

def load_faiss(model, documents, name):
    if path.exists(f"models/{name}.index"):
        return faiss.read_index(f"models/{name}.index")
    else:
        return build_faiss(model, documents, name)
        

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
