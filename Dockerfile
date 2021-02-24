FROM python:3.8

RUN pip install pandas numpy strawberry-graphql[debug-server] nltk scipy scikit-learn faiss-cpu dnspython cache_to_disk pymongo sentence-transformers

WORKDIR /code
COPY . /code

ENTRYPOINT strawberry server app
