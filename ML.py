from textgenie import TextGenie
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest 
from transformers import pipeline

def paraphraser(docs):
    nlp =spacy.load("en_core_web_sm")
    doc= nlp(docs)
    sentences = [sent.text.strip() for sent in doc.sents]

    textgenie = TextGenie("hetpandya/t5-small-tapaco", "bert-base-uncased")
    result="" 

    for i in sentences:
        res=textgenie.augment_sent_t5(i,"paraphrase: ",max_length=200, n_predictions=1)
        result= result+res[0]
    return result


#res=paraphraser( "One morning, when Gregor Samsa woke from troubled dreams, he found himself transformed in his bed into a horrible vermin. He lay on his armour-like back, and if he lifted his head a little he could see his brown belly, slightly domed and divided by arches into stiff sections. The bedding was hardly able to cover it and seemed ready to slide off any moment. His many legs, pitifully thin compared with the size of the rest of him, waved about helplessly as he looked. ")
#print(res)
def summarize(doc):
    nlp = spacy.load("en_core_web_sm")
    doc= nlp(doc)
    keyword=[]
    stopwords= list(STOP_WORDS)
    pos_tag= ["PROPN","ADJ","NOUN", "VERB"]
    for token in doc:
        if(token.text in stopwords or token.text in punctuation):
            continue
        if (token.pos_ in pos_tag):
            keyword.append(token.text)
    
    freq_word= Counter(keyword)
    freq_word.most_common(5)
    max_freq= Counter(keyword).most_common(1)[0][1]
    for word in freq_word.keys():
        freq_word[word]= (freq_word[word]/max_freq)
    freq_word.most_common(5)

    

    sent_strength={}
    for sent in doc.sents:
        for word in sent:
            if word.text in freq_word.keys():
                if sent in sent_strength.keys():
                    sent_strength[sent]+=freq_word[word.text]
                else:
                    sent_strength[sent]=freq_word[word.text]
    #print(sent_strength)

    summarized_sentences= nlargest(3, sent_strength, key=sent_strength.get)

    final_sentences= [w.text for w in summarized_sentences]
    summary= " ".join(final_sentences)
    return summary


def generator(prompt):
    generator = pipeline('text-generation', model ='EleutherAI/gpt-neo-125M')
    return generator(prompt, max_length=200, do_sample=True, temperature=0.9)[0]["generated_text"]

#res=generator("One morning, when Gregor Samsa woke from troubled dreams.")
#print(res)
#result=textToImage("A football player cartoon")