import strawberry
from strawberry.flask.views import GraphQLView
from flask import Flask
from typing import List, Dict, Any
from search import *
from build import *
from utils import docs2text, id2details
from cloud_storage import test_file_exists, download_blob, upload_blob, pull_indices, download_pytorch_model
from preprocessing import preprocess_QA_text, preprocess_string, ensure_good_content, ensure_good_str_list, ensure_good_string, ensure_good_list
from bson import ObjectId
from live_indexing import update_faiss

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
class Author:
    url: str
    name: str
    email: str

@strawberry.type
class JournalReference:
    title: str
    volume: str
    issue: str
    start: str
    end: str

@strawberry.type
class Image:
    url: str
    description: str
    provider: str
    license: str

@strawberry.type
class Source:
    id: str
    name: str
    description: str
    url: str

@strawberry.type
class Document:
    id: str
    url: str
    directURL: str
    title: str
    dateIndexed: str
    type_: str
    content: List[str]
    alternateTitle: List[str]
    fileName: str
    authors: List[Author]
    datePublished: str
    keywords: List[str]
    description: str
    alternateDescription: str
    imageURLS: List[Image]
    isbn: str
    issn: str
    doi: str
    pubMedID: str
    pmcID: str
    publisher: str
    journalReference: JournalReference
    meshHeadings: List[str]
    meshQualifiers: List[str]
    source: Source
    rights: str
    language: str

@strawberry.type
class SearchResult:
    documents: List[Document]

@strawberry.type
class QAResult:
    answer: str
    confidence : float


@strawberry.type
class MetaData:
    tf_idf_len_diff: int
    bert_len_diff: int

@strawberry.type
class IndexingResult:
    status: str
    metadata: MetaData

def serach_result_from_documents(documents):
    return SearchResult(
        [Document(
            id = doc['_id'],  #ensure_good_string(doc, '_id'),
            url = doc['url'], #ensure_good_string(doc,'url'),
            directURL = doc['directURL'], #ensure_good_string(doc,'directURL'),
            title = doc['title'].encode('latin1').decode('utf8'),
            dateIndexed = doc['dateIndexed'], #ensure_good_string(doc,'dateIndexed'),
            type_ = doc['type'], #ensure_good_string(doc,'type'),
            content = ensure_good_content(doc['content']['text']), # note difference!
            alternateTitle = ensure_good_str_list(doc,'alternateTitle'),
            fileName = ensure_good_string(doc,'file_name'),
            authors = [
                Author(
                    url = ensure_good_string(author,'url'),
                    name = ensure_good_string(author,'name'),
                    email = ensure_good_string(author,'email')
                )
                for author in ensure_good_list(doc, 'authors')],
            datePublished = ensure_good_string(doc,'datePublished'),
            keywords = ensure_good_str_list(doc,'keywords'),
            description = ensure_good_string(doc,'description'),
            alternateDescription = ensure_good_string(doc,'alternateDescription'),
            imageURLS = [
                Image(
                    url = ensure_good_string(image,'url'),
                    description = ensure_good_string(image,'description'),
                    provider = ensure_good_string(image,'provider'),
                    licence = ensure_good_string(image,'licence')
                )
                for image in ensure_good_list(doc, 'imageURLs')],
            isbn = ensure_good_string(doc,'isbn'),
            issn = ensure_good_string(doc,'issn'),
            doi=ensure_good_string(doc, 'doi'),
            pubMedID=ensure_good_string(doc, 'pubMedID'),
            pmcID=ensure_good_string(doc, 'pmcID'),
            publisher=ensure_good_string(doc, 'publisher'),
            journalReference=JournalReference(
                title=ensure_good_string(doc.get('journalReference', {}), 'title'),
                volume=ensure_good_string(doc.get('journalReference', {}), 'volume'),
                issue=ensure_good_string(doc.get('journalReference', {}), 'issue'),
                start=ensure_good_string(doc.get('journalReference', {}), 'start'),
                end=ensure_good_string(doc.get('journalReference', {}), 'end'),
            ),
            meshHeadings = ensure_good_str_list(doc,'meshHeadings'),
            meshQualifiers = ensure_good_str_list(doc,'meshQualifiers'),
            source = Source(
                id = ensure_good_string(doc['source'],'id'),
                name = ensure_good_string(doc['source'],'name'),
                description = ensure_good_string(doc['source'],'description'),
                url = ensure_good_string(doc['source'],'url')
            ), #ensure_good_string(doc,'source'),
            rights = ensure_good_string(doc,'rights'),
            language = doc['language'], #ensure_good_string(doc,'language')
        ) for doc in documents])

@strawberry.type
class Query:

    @strawberry.field
    def search(self, q: str, language: str = 'en', type: str = None, limit: int = 20) -> SearchResult:

        D1, I1 = vector_search(q, tfidf_model, tfidf_faiss, k = limit)
        D2, I2 = vector_search(q, bert_model, bert_faiss, k = limit)

        combined_results = combine_results(D1, I1, D2, I2)

        id_arr = (np.array(ids)[combined_results]).tolist()
        # print("New: {}".format(id_arr))
        # I = combined_results
        filters = {'language':language, '_id': {'$in': id_arr}}
        if not type is None:
            filters['type'] = type
        documents = list(collection.find(filters))

        # things = list(db.things.find({'_id': {'$in': id_array}}))
        documents.sort(key=lambda doc: id_arr.index(doc['_id']))
        return serach_result_from_documents(documents)


    @strawberry.field
    def more_docs(self, id: str) -> SearchResult:
        doc = list(collection.find({'_id': ObjectId(id)}))[0]
        D, I = vector_search(doc['title']+doc['description'], bert_model, bert_faiss, k = 10)
        id_arr = (np.array(ids)[I[0]]).tolist()
        documents = list(collection.find({'_id': {'$in': id_arr}}))

        return serach_result_from_documents(documents)

    # @strawberry.field
    # def semantic_search(self, q: str) -> SearchResult:
    #     D, I = vector_search(q, bert_model, bert_faiss)
    #     return SearchResult([Document(id=doc['_id'], title=doc['title'],description=doc['description']) for _, doc in id2details(documents, I).iterrows()])

    @strawberry.field
    def qa(self, q: str) -> QAResult:
        # Find top 2 relevant documents
        D, I = vector_search(q, bert_model, bert_faiss)
        reference = [x["description"] for x in collection.find({'_id': {'$in': (np.array(ids)[I[0][:3]]).tolist()},  'source.id': {'$in' : ["nhs_az", "nhs_med", "nhs_covid19", "bnf"]} })]

        # preprocess docs before search
        qa_clean_q = preprocess_string(q, lowercasing=False, stemming=False, stopping=False)
        print(f"Raw references in QA: {reference}")
        clean_refs = [preprocess_QA_text(ref) for ref in reference]
        print(f"Cleaned references to search in: {clean_refs}")
        # answer = qa_model.predict(" ".join(clean_refs), qa_clean_q)
        answer = qa_model.predict(" ".join(clean_refs), qa_clean_q)

        return QAResult(answer = answer['answer'], confidence = answer['confidence'])

    @strawberry.field
    def pull_updates_from_index_cloud(self) -> IndexingResult:
        global tfidf_faiss, bert_faiss, ids
        tf_idf_prev_len = tfidf_faiss.ntotal
        bert_prev_len = bert_faiss.ntotal

        print("Previous TFIDF length: {}".format(tf_idf_prev_len))
        pull_indices(True)

        tfidf_faiss,  bert_faiss = load_faiss(tfidf_model, bert_model)
        ids = load('models/ids.joblib')

        metadata = MetaData(tf_idf_len_diff = tfidf_faiss.ntotal - tf_idf_prev_len, bert_len_diff = bert_faiss.ntotal - bert_prev_len)
        return IndexingResult(status = "Success", metadata = metadata)

    if os.environ.get('REINDEXING_INSTANCE') != None:
        @strawberry.field
        def reindex(self) -> IndexingResult:
            global bert_model, tfidf_model, tfidf_faiss, bert_faiss, ids
            tf_idf_prev_len = tfidf_faiss.ntotal
            bert_prev_len = bert_faiss.ntotal

            print("Previous TFIDF length: {}".format(tf_idf_prev_len))


            tfidf_faiss, bert_faiss, ids = update_faiss(tfidf_model, bert_model, tfidf_faiss, bert_faiss, ids)
            metadata = MetaData(tf_idf_len_diff = tfidf_faiss.ntotal - tf_idf_prev_len, bert_len_diff = bert_faiss.ntotal - bert_prev_len)

            return IndexingResult(status = "Success", metadata = metadata)

schema = strawberry.Schema(query=Query)
app = Flask(__name__)
app.add_url_rule('/graphql', view_func=GraphQLView.as_view('graphql_view', schema=schema))
