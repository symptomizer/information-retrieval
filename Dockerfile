FROM python:3.8

RUN pip install -r requirements.txt
RUN python install_ci.py

WORKDIR /code
COPY . /code
COPY ./nltk_data /usr/local/nltk_data

EXPOSE 8000

ENTRYPOINT strawberry server app
