FROM python:3.8

COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

WORKDIR /code
COPY . /code
COPY ./nltk_data /usr/local/nltk_data
ENV PYTHONUNBUFFERED=1

ENTRYPOINT strawberry server app
