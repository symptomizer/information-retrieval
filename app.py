import strawberry
from typing import List
from search import tfidf_search
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
        return SearchResult([Document(id="asd", name=doc[0],description=doc[1]) for doc in tfidf_search(q)])


schema = strawberry.Schema(query=Query)
