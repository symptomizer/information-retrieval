import strawberry
from typing import List
from search import *
from build import *
from utils import docs2text, id2details
# from deeppavlov import build_model
# dp_model = build_model('models/squad_torch_bert.json', download=True)

# documents = get_data()

# bert_model = build_bert_model()
# text = docs2text(documents)
bert_model = load_bert_model()
tfidf_model = load_tfidf_model()
tfidf_faiss,  bert_faiss = load_faiss(tfidf_model, bert_model)
ids = load('models/ids.joblib')
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
    content: List[str]
    url: str
    source: str
    language: str

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
    def search(self, q: str, language: str = 'en') -> SearchResult:
        D1, I1 = vector_search(q, tfidf_model, tfidf_faiss)
        D2, I2 = vector_search(q, bert_model, bert_faiss)
        # D2, I2 = vector_search(q, tfidf_model, tfidf_faiss)

        I = list(set([x[1] for x in sorted((list(zip(D1[0],I1[0])) + list(zip(D2[0],I2[0]))), key = lambda x:x[0])]))

        documents = collection.find({'_id': {'$in': (np.array(ids)[I[:10]]).tolist()}, 'language': language})
        return SearchResult([Document(id=doc['_id'], title=doc['title'].encode('latin1').decode('utf8'),description=doc['description'], content=doc['content']['text'], url=doc['directURL'], source=doc['source']['id'], language=doc['language']) for doc in documents])
    # @strawberry.field
    # def semantic_search(self, q: str) -> SearchResult:
    #     D, I = vector_search(q, bert_model, bert_faiss)
    #     return SearchResult([Document(id=doc['_id'], title=doc['title'],description=doc['description']) for _, doc in id2details(documents, I).iterrows()])

    @strawberry.field
    def qa(self, q: str) -> QAResult:
        D, I = vector_search(q, bert_model, bert_faiss)
        reference = [x["description"] for x in collection.find({'_id': {'$in': (np.array(ids)[I[0][:2]]).tolist()}})]
        answer = qa_model.predict(" ".join(reference),q)
        return QAResult(answer = answer['answer'], confidence = answer['confidence'], )

schema = strawberry.Schema(query=Query)
