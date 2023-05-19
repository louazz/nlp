import json
from flask import Flask, jsonify,request, send_file

from ML import summarize, paraphraser, textToImage

app = Flask(__name__)


@app.route('/summarize', methods=['POST'])
async def summarizer():
    doc=json.loads(request.data)["data"]
    res= summarize(doc)
    return jsonify({"data": res})

@app.route("/paraphrase", methods=['POST'])
async def paraphrase():
    doc=json.loads(request.data)["data"]
    res=  paraphraser(doc)
    return jsonify({"data": res})

@app.route("/image", methods=["POST"])
async def image():
    doc=json.loads(request.data)["data"]
    res= textToImage(doc)
    return send_file(res, mimetype="image/png")


app.run()
