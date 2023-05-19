import json
from flask import Flask, jsonify,request
from flask_cors import CORS
from ML import generator, summarize, paraphraser
from flask_mail import Mail, Message

app = Flask(__name__)
CORS(app)
mail= Mail(app)

app.config['MAIL_SERVER']='smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'encrygen@gmail.com'
app.config['MAIL_PASSWORD'] = 'cqxyjjvtiurhkfkj'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
mail = Mail(app)

@app.route("/send", methods=["POST"])
def index():
   email=json.loads(request.data)["email"]
   feedback=json.loads(request.data)["feedback"]
   msg = Message('feedback', sender = email, recipients = ['encrygen@gmail.com'])
   msg.body = feedback
   mail.send(msg)
   return "Sent"

@app.route('/summarize', methods=['POST'])
async def summarizer():
    doc=json.loads(request.data)["data"]
    print(json.loads(request.data))
    res= summarize(doc)
    return jsonify({"data": res})

@app.route("/paraphrase", methods=['POST'])
async def paraphrase():
    doc=json.loads(request.data)["data"]
    res=  paraphraser(doc)
    return jsonify({"data": res})


@app.route("/generate", methods=['POST'])
async def generate():
    doc=json.loads(request.data)["data"]
    res=  generator(doc)
    return jsonify({"data": res})



if __name__ == '__main__':
   app.run(debug = True)
