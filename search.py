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

"""Extract nested values from a JSON tree."""


def json_extract(obj, key):
    """Recursively fetch values from nested JSON."""
    arr = []

    def extract(obj, arr, key):
        """Recursively search for values of key in JSON tree."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    extract(v, arr, key)
                elif k == key:
                    arr.append(v)
        elif isinstance(obj, list):
            for item in obj:
                extract(item, arr, key)
        return arr

    values = extract(obj, arr, key)
    return values


def get_nhs_data(path="NHS/A-Z/nhs_az.json"):
    with open(path) as json_file:
        data = json.load(json_file)
        print(type(list(data.values())[0]))
        df  = pd.DataFrame([json.loads(obj) for obj in list(data.values())])
        df["text"] = df["mainEntityOfPage"].apply(lambda x: " ".join(json_extract(x,"text")))
    return df.astype(str)


def preprocess(x):
    return " ".join(re.sub(r'[^a-zA-Z]',' ',w).lower() for w in x.split() if re.sub(r'[^a-zA-Z]',' ',w).lower() not in stopwords.words('english'))


nhs_df = get_nhs_data()



tfidf_vectoriser=TfidfVectorizer(ngram_range=(3,5), analyzer="char")
datasetTFIDF = tfidf_vectoriser.fit_transform(nhs_df['description']+nhs_df['name']+nhs_df['text'])

def tfidf_search(q):

    queryTFIDF = tfidf_vectoriser.transform([q])
    cosine_similarities = cosine_similarity(queryTFIDF, datasetTFIDF).flatten()
    print([[row["name"],row["description"]] for i,row in nhs_df.iloc[cosine_similarities.argsort()[:-11:-1]].iterrows()])
    return [[row["name"],row["description"]] for i,row in nhs_df.iloc[cosine_similarities.argsort()[:-11:-1]].iterrows()]
