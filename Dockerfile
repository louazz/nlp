FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install gunicorn flask spacy torch transformers textgenie flask_cors

ENTRYPOINT ["sh","./gunicorn.sh"]

