FROM python:3.8

RUN pip install pandas numpy strawberry-graphql[debug-server] nltk scipy scikit-learn

WORKDIR /code
COPY . /code

ENTRYPOINT strawberry server app