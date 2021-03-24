import strawberry
from typing import List, Dict, Any
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
    rights: str
    url: str
    language: str
    type: str
    directURL: str
    datePublished: str
    dateAdded: str

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
    def search(self, q: str, language: str = 'en', type: str = None) -> SearchResult:
        number_of_results = 20

        D1, I1 = vector_search(q, tfidf_model, tfidf_faiss, k = number_of_results)
        D2, I2 = vector_search(q, bert_model, bert_faiss, k = number_of_results)
        # D2, I2 = vector_search(q, tfidf_model, tfidf_faiss)
        combined_results = combine_results(D1, I1, D2, I2)
        # I = list(set([x[1] for x in sorted((list(zip(0.25*D1[0],I1[0])) + list(zip(0.75*D2[0],I2[0]))), key = lambda x:x[0])]))
        I = list(
                set(
                    [x[1] for x in sorted(
                        (list(zip(0.25*D1[0],I1[0])) + list(zip(0.75*D2[0],I2[0]))), key = lambda x:x[0])
                        ]
                    )
                )
        print(I)
        filters = {'language':language, '_id': {'$in': (np.array(ids)[I[:10]]).tolist()}}
        if not type is None:
            filters['type'] = type
        documents = collection.find( filters)
        return SearchResult([Document(id=doc['_id'], title=doc['title'].encode('latin1').decode('utf8'),description=doc['description'], content=doc['content']['text'], url=doc['url'],directURL=doc['directURL'], type=doc['type'], language=doc['language'], rights=doc['rights'], datePublished=doc['datePublished'], dateAdded=doc['dateIndexed']) for doc in documents])
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
