import strawberry
from typing import List, Dict, Any
from search import *
from build import *
from utils import docs2text, id2details
from cloud_storage import test_file_exists, download_blob, upload_blob, pull_indices, download_pytorch_model
from preprocessing import preprocess_QA_text, preprocess_string, ensure_good_content
from bson import ObjectId
# from deeppavlov import build_model
# dp_model = build_model('models/squad_torch_bert.json', download=True)

# documents = get_data()

# bert_model = build_bert_model()
# text = docs2text(documents)

# GCP test connections
test_file_exists()
download_blob("symptomizer_indices_bucket-1", "hello.txt", "test.txt")
download_pytorch_model()
pull_indices()

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

        combined_results = combine_results(D1, I1, D2, I2)
       
        id_arr = (np.array(ids)[combined_results]).tolist()
        # print("New: {}".format(id_arr))
        # I = combined_results
        filters = {'language':language, '_id': {'$in': id_arr}}
        if not type is None:
            filters['type'] = type
        documents = list(collection.find(filters))


        # def ensure_good_content(content_list):
        #     '''
        #     function to remove potential problems from the context, and preprocess it to look like normal text
        #     '''
        #     # remove None-s from the list
        #     string_list = map(str,content_list)
        #     # preprocess and join together the content list
        #     string_list = [preprocess_string(page, stopping = False, stemming = False, lowercasing = False) for page in string_list ]
        #     return [string_list]

        # things = list(db.things.find({'_id': {'$in': id_array}}))
        documents.sort(key=lambda doc: id_arr.index(doc['_id']))
        return SearchResult([Document(id=doc['_id'], 
        title=doc['title'].encode('latin1').decode('utf8'),
        description=doc['description'], 
        content= ensure_good_content(doc['content']['text']),
        url=doc['url'],
        directURL=doc['directURL'], 
        type=doc['type'], 
        language=doc['language'], 
        rights="", 
        datePublished=doc['datePublished'], 
        dateAdded=doc['dateIndexed']) for doc in documents])

    @strawberry.field
    def more_docs(self, id: str) -> SearchResult:
        doc = list(collection.find({'_id': ObjectId(id)}))[0]
        D, I = vector_search(doc['title']+doc['description'], bert_model, bert_faiss, k = 10)
        id_arr = (np.array(ids)[I[0]]).tolist()
        documents = list(collection.find({'_id': {'$in': id_arr}}))
        
        return SearchResult([Document(id=doc['_id'], 
        title=doc['title'].encode('latin1').decode('utf8'),
        description=doc['description'], 
        content= ensure_good_content(doc['content']['text']),
        url=doc['url'],
        directURL=doc['directURL'], 
        type=doc['type'], 
        language=doc['language'], 
        rights="", 
        datePublished=doc['datePublished'], 
        dateAdded=doc['dateIndexed']) for doc in documents])
    # @strawberry.field
    # def semantic_search(self, q: str) -> SearchResult:
    #     D, I = vector_search(q, bert_model, bert_faiss)
    #     return SearchResult([Document(id=doc['_id'], title=doc['title'],description=doc['description']) for _, doc in id2details(documents, I).iterrows()])

    @strawberry.field
    def qa(self, q: str) -> QAResult:
        # Find top 2 relevant documents
        D, I = vector_search(q, bert_model, bert_faiss)
        reference = [x["description"] for x in collection.find({'_id': {'$in': (np.array(ids)[I[0][:3]]).tolist()},  'source.id': "nhs_az"})]

        # preprocess docs before search
        qa_clean_q = preprocess_string(q, lowercasing=False, stemming=False, stopping=False)
        print(f"Raw references in QA: {reference}")
        clean_refs = [preprocess_QA_text(ref) for ref in reference]
        print(f"Cleaned references to search in: {clean_refs}")
        # answer = qa_model.predict(" ".join(clean_refs), qa_clean_q)
        answer = qa_model.predict(" ".join(clean_refs), qa_clean_q)

        return QAResult(answer = answer['answer'], confidence = answer['confidence'])

schema = strawberry.Schema(query=Query)
