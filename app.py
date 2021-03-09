import strawberry
from typing import List
from search import *
from build import *
from utils import docs2text, id2details
# from deeppavlov import build_model
# dp_model = build_model('models/squad_torch_bert.json', download=True)

documents = get_data()

bert_model = build_bert_model()
text = docs2text(documents)
tfidf_model = build_tfidf_model(text)
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
    content: List[str]
    url: str
    source: str

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
        D1, I1 = vector_search(q, tfidf_model, tfidf_faiss)
        D2, I2 = vector_search(q, bert_model, bert_faiss)

        I = list(set([x[1] for x in sorted((list(zip(D1[0],I1[0])) + list(zip(D2[0],I2[0]))), key = lambda x:x[0])]))
        
        return SearchResult([Document(id=doc['_id'], title=doc['title'],description=doc['description'], content=doc['content']['text'], url=doc['directURL'], source=doc['source']['id']) for _, doc in id2details(documents, I[:10]).iterrows()])
    # @strawberry.field
    # def semantic_search(self, q: str) -> SearchResult:
    #     D, I = vector_search(q, bert_model, bert_faiss)
    #     return SearchResult([Document(id=doc['_id'], title=doc['title'],description=doc['description']) for _, doc in id2details(documents, I).iterrows()])

    @strawberry.field
    def qa(self, q: str) -> QAResult:
        D, I = vector_search(q, tfidf_model, tfidf_faiss)
        # reference = ""
        # for _, doc in docs2text(id2details(documents, I[0])).iterrows():
        #     print(doc)
        #     reference += " " + doc
        # print(docs2text(id2details(documents, I[0])))
        answer = qa_model.predict(" ".join(documents['description'].iloc[I[0][0:2]]),q)
        return QAResult(answer = answer['answer'], confidence = answer['confidence'], )

schema = strawberry.Schema(query=Query)
