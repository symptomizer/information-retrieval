import pandas as pd
import numpy as np
import nltk
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from build import *
import faiss


def vector_search(q, model, index):

    vector = []
    if hasattr(model, 'encode'):
        vector = model.encode([q])
    else:
        vector = model.transform([q]).toarray()
    vector = vector.astype("float32")
    faiss.normalize_L2(vector)
    D, I = index.search(vector, k=10)
    return D, I

    # return [list(df[df._id == idx][column]) for idx in I[0]]
