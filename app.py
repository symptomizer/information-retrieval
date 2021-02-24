import strawberry
from typing import List
from search import *
from build import *

documents = get_data()
bert_model = build_bert_model()
text = documents['description']+documents['title']+documents['content'].apply(lambda arr: " ".join(arr))
tfidf_model = build_tfidf_model(documents['description']+documents['title'])
tfidf_faiss = load_faiss(tfidf_model, text, "tfidf")
bert_faiss = load_faiss(bert_model, text, "bert")


@strawberry.type
class User:
    name: str
    age: int

@strawberry.type
class Document:
    id: str
    title: str
    description: str

@strawberry.type
class SearchResult:
    documents: List[Document]

@strawberry.type
class Query:
    @strawberry.field
    def search(self, q: str) -> SearchResult:
        D, I = vector_search(q, tfidf_model, tfidf_faiss)
        return SearchResult([Document(id=doc['_id'], title=doc['title'],description=doc['description']) for _, doc in id2details(documents, I).iterrows()])
    @strawberry.field
    def semantic_search(self, q: str) -> SearchResult:
        D, I = vector_search(q, bert_model, bert_faiss)
        return SearchResult([Document(id=doc['_id'], title=doc['title'],description=doc['description']) for _, doc in id2details(documents, I).iterrows()])


schema = strawberry.Schema(query=Query)
