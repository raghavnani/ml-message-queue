FROM python:3.7-buster

COPY requirements.txt /src/

WORKDIR /src

RUN apt-get update -y && \
    apt-get upgrade -y && \
    pip install -r requirements.txt

COPY nlp .

COPY message_queue.py .

CMD celery -A message_queue worker --pool threads --loglevel=info
