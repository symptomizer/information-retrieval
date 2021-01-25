import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

def preprocess(x):
    return " ".join(re.sub(r'[^a-zA-Z]',' ',w).lower() for w in x.split() if re.sub(r'[^a-zA-Z]',' ',w).lower() not in stopwords.words('english'))


with open('example_data/NHS/health_az_a.json') as json_file:
    data = json.load(json_file)
    df  = pd.DataFrame(data['significantLink'])

tfidf_vectoriser=TfidfVectorizer(ngram_range=(1,2), analyzer="char")
datasetTFIDF = tfidf_vectoriser.fit_transform(df['description']+df['name'])

def tfidf_search(q):
    queryTFIDF = tfidf_vectoriser.transform([q])
    cosine_similarities = cosine_similarity(queryTFIDF, datasetTFIDF).flatten()
    print([[row["name"],row["description"]] for i,row in df.iloc[cosine_similarities.argsort()[:-11:-1]].iterrows()])
    return [[row["name"],row["description"]] for i,row in df.iloc[cosine_similarities.argsort()[:-11:-1]].iterrows()]
