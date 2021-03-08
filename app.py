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
qa_model = QA('models')

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
class QAResult:
    answer: str
    confidence : float

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

    @strawberry.field
    def qa(self, q: str) -> QAResult:
        D, I = vector_search(q, tfidf_model, tfidf_faiss)
        reference = ""
        for _, doc in id2details(documents, I).iterrows():
            reference += " " + doc["description"]
        answer = qa_model.predict(reference,q)
        print(answer['document'])
        return QAResult(answer = answer['answer'], confidence = answer['confidence'], )

schema = strawberry.Schema(query=Query)
