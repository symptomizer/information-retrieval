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


def vector_search(q, model, index, k=10):

    vector = []
    if hasattr(model, 'encode'):
        vector = model.encode([q])
    else:
        vector = model.transform([q]).toarray()
    vector = vector.astype("float32")
    faiss.normalize_L2(vector)
    D, I = index.search(vector, k=k)
    return D, I

    # return [list(df[df._id == idx][column]) for idx in I[0]]


def combine_results(D1_np, I1_np, D2_np, I2_np):
    """
    D1: Scores for TFIDF
    I1: IDs for TFIDF
    D2: Scores for Bert
    I2: IDs for Bert
    """
    D1 = D1_np[0]
    I1 = I1_np[0]
    D2 = D2_np[0]
    I2 = I2_np[0]
    combined_zips = []

    zipped_tf_idf = list(zip(I1, 0.25*D1))
    zipped_bert   = list(zip(I2, 0.75*D2))
    # clean_bert = []
    # print("Before Combined:")
    # print(zipped_tf_idf)
    # print(zipped_bert)
    for doc_ind, score in zipped_tf_idf:
        if (doc_ind in I2):
            i = np.where(I2 == doc_ind)[0][0]
            _ , bert_score = zipped_bert[i]
            combined_zips.append((doc_ind, score + bert_score))
        else:
            combined_zips.append((doc_ind, score))
 
    for doc_ind, score in zipped_bert:
        if (not doc_ind in I1):
            combined_zips.append((doc_ind, score))


    # print("After Combined:")
    # print(combined_zips)
    sorted_results_with_scores = [doc_score_tup for doc_score_tup in sorted(combined_zips, key = lambda x:x[1], reverse=True)]
    print(sorted_results_with_scores)
    sorted_results = [doc_score_tup[0] for doc_score_tup in sorted(combined_zips, key = lambda x:x[1], reverse=True)]
    # print("Sorted Combined:")
    # print(sorted_results)

    return sorted_results
    