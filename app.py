import strawberry
from typing import List

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

@strawberry.type
class User:
    name: str
    age: int

@strawberry.type
class Document:
    id: str
    name: str
    description: str

@strawberry.type
class SearchResult:
    documents: List[Document]


@strawberry.type
class Query:
    @strawberry.field
    def search(self, q: str) -> SearchResult:
        return tfidf_search(q)


def preprocess(x):
    return " ".join(re.sub(r'[^a-zA-Z]',' ',w).lower() for w in x.split() if re.sub(r'[^a-zA-Z]',' ',w).lower() not in stopwords.words('english'))


with open('example_data/NHS/health_az_a.json') as json_file:
    data = json.load(json_file)
    df  = pd.DataFrame(data['significantLink'])

tfidf_vectoriser=TfidfVectorizer()
datasetTFIDF = tfidf_vectoriser.fit_transform(df['description'].apply(lambda x: preprocess(x))+df['name'].apply(lambda x: preprocess(x)))

def tfidf_search(q):
    queryTFIDF = tfidf_vectoriser.transform([preprocess(q)])
    cosine_similarities = cosine_similarity(queryTFIDF, datasetTFIDF).flatten()
    results = []
    for index, row in df.iloc[cosine_similarities.argsort()[:-11:-1]].iterrows():
        results.append(Document(id="sdf",name=row["name"], description=row["description"]))
    return SearchResult(documents=results)


schema = strawberry.Schema(query=Query)
