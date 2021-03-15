# from deeppavlov import build_model
from build import *
import sys
from joblib import dump, load
# import dask.dataframe as dd
# import pymongo
# from sklearn.feature_extraction.text import TfidfVectorizer
# # from blaze import TableSymbol, compute
# # documents = TableSymbol('documents', '{title: string, description: string}')
collection = pymongo.MongoClient('mongodb+srv://ir:UE5Ki3Cr1eyWVeNl@cluster1.xo9vl.mongodb.net/document?retryWrites=true&w=majority').document.document
 
# TfidfVectorizer().fit(x['title'] or "" + " " + x['description'] or "" + " " + " ".join(filter(None,x['content']['text'] or [])) for x in collection.find({}, projection={'title': True, 'description': True, "content.text": True}).limit(300))

# # model = build_model('models/squad_torch_bert.json', download=True)
# model("When was James born?","James was born in 1977.")
# model(["James was born in 1977."],["When was James born?"]) 

# print(list((x.get("title","") + " " + x.get('description',"")+ " " + " ".join(filter(None,x['content']['text'] or [])) for x in collection.find({}, projection={'title': True, 'description': True, "content.text": True}).limit(3))))

# get_data()
# print('hi')
# df = dd.read_parquet("models/documents")
# print(df)

# print("downloaded")
# # l = list(x for x in collection.find().skip(1).limit(1))
# # with open('text.txt', 'w') as f:
# #     for item in l[0]['content']['text']:
# #         f.write("%s\n" % item.encode('latin1').decode('utf-8'))
# model = load_tfidf_model()
# build_faiss(model, "tfidf")
ids = load('models/ids.joblib')
print(ids)
