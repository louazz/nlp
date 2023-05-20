
FROM python:3.8-slim-buster

WORKDIR /python-docker
RUN apt-get -y update && apt-get install -y git
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install gunicorn flask

COPY . .

ENTRYPOINT ["sh","./gunicorn.sh"]

