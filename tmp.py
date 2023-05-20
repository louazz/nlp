
import spacy
nlp =spacy.load("en_core_web_sm")
doc = nlp("He lay on his armour-like back, and if he lifted his head a little he could see his brown belly, slightly domed and divided by arches into stiff sections. One morning, when Gregor Samsa woke from troubled dreams, he found himself transformed in his bed into a horrible vermin. His many legs, pitifully thin compared with the size of the rest of him, waved about helplessly as he looked.")
sentences = [sent.text.strip() for sent in doc.sents]
textgenie = TextGenie("hetpandya/t5-small-tapaco", "bert-base-uncased")

# Augment a list of sentences
sentences =sentences
res=textgenie.magic_lamp(
    sentences, "paraphrase: ", n_mask_predictions=5, convert_to_active=True
)
result=""
for i in res:
    result= result+i
print(result)