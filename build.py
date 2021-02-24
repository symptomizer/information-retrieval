import nltk
from collections import defaultdict
from nltk.stem.snowball import EnglishStemmer  # Assuming we're working with English
import pymongo
import faiss
import pandas as pd
import numpy as np
# nltk.download('stopwords')
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from cache_to_disk import cache_to_disk
from cache_to_disk import delete_disk_caches_for_function
from sentence_transformers import SentenceTransformer
from os import path
# delete_disk_caches_for_function('get_data')


@cache_to_disk(2)
def get_data():
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
    model = SentenceTransformer('paraphrase-distilroberta-base-v1')
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
    index = faiss.IndexFlatL2(embeddings.shape[1])

    # Step 3: Pass the index to IndexIDMap
    # index = faiss.IndexIDMap(index)
    # Step 4: Add vectors and their IDs
    index.add(embeddings)
    faiss.write_index(index,f"{name}.index")
    print(f"Completed {name} index.")
    return index
    

def load_faiss(model, documents, name):
    if path.exists(f"{name}.index"):
        return faiss.read_index(f"{name}.index")
    else:
        return build_faiss(model, documents, name)

 
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
