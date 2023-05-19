import spacy 
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest 
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from diffusers import StableDiffusionPipeline


def paraphraser(doc):
    input_sentence=doc
    model = BartForConditionalGeneration.from_pretrained('eugenesiow/bart-paraphrase')
    device = torch.device = torch.device('cpu')
    model= model.to(device)
    tokenizer= BartTokenizer.from_pretrained('eugenesiow/bart-paraphrase')
    batch= tokenizer(input_sentence, return_tensors='pt')
    generated_ids= model.generate(batch['input_ids'])
    generated_sentence= tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_sentence




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


def textToImage(text):
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe= StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe= pipe.to("cuda")

    prompt= text 
    image= pipe(prompt).image[0]
    image.save("tmp.png")
    return image

result=textToImage("A football player cartoon")