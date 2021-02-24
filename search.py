import pandas as pd
import numpy as np
import nltk
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from build import *


def vector_search(q, model, index):

    vector = []
    if hasattr(model, 'encode'):
        vector = model.encode([q])
    else:
        vector = model.transform([q]).toarray()

    D, I = index.search(vector.astype("float32"), k=10)
    return D, I

def id2details(df, I, columns=None):
    """Returns the paper titles based on the paper index."""
    if columns is not None:
        return df.iloc[I[0]][columns]
    return df.iloc[I[0]]
    # return [list(df[df._id == idx][column]) for idx in I[0]]
