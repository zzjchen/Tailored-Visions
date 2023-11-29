# Code by zzjchen
# This code includes functions for 2 types of Retriever
import spacy
import os
from utils import read_jsonl_data,read_multi_jsonl_file
import re
import torch
import clip
from metrics import calculate_bleu
import numpy as np
from collections import Counter
from nltk.tokenize import word_tokenize
def remove_specific_characters(text):
    text_without_chars = re.sub(r'[@#{}[\]()+-]', '', text)
    return text_without_chars

def noun_extraction(nlp,text):
    doc = nlp(text)
    nouns=[chunk.text for chunk in doc.noun_chunks]
    ents=[entity.text for entity in doc.ents]
    return nouns,ents

@torch.no_grad()
def clip_similarity(model, query, texts):
    '''
    Calculating CLIP similarity score between query and texts.
    Note:
        We only need to rank top-k texts most similar to query, thus we only user 'score'.

    Args:
        model: CLIP model
        query: Query
        texts: List of texts

    Returns:
        Dict including:
            'score': clip similarity results
            'similarity': softmax probs
    '''

    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_new = clip.tokenize(query,truncate=True).to(device)
    text_words=clip.tokenize(texts, truncate=True).to(device)
    with torch.no_grad():
        new_features = model.encode_text(text_new)
        words_features=model.encode_text(text_words)
        words_distance=(words_features-new_features).norm(dim=1)
        new_features/=new_features.norm(dim=-1,keepdim=True)
        words_features/=words_features.norm(dim=-1,keepdim=True)
        words_score=( 100*new_features @ words_features.T)
        words_similarity=words_score.softmax(dim=1)
        if len(words_score.shape)==2:
            words_score=words_score.squeeze(0)
            words_similarity=words_similarity.squeeze(0)
        
        
    return {"score":words_score,"similarity":words_similarity,}

def checklines(lines,threshold=0.5):
    '''
    Check if prompts in lines are very close to each other
    Args:
        lines: List of dicts, each represents a T2I history
        threshold: threshold for bleu score

    Returns:
        Unsilimar T2I histories.
    '''
    references=[]
    image_urls=[]
    for line in lines:
        if references==[]:
            references.append(line['prompt'])
            image_urls.append(line['result_url'])
        else:
            hypothesis=line['prompt']
            if calculate_bleu(references,hypothesis)<threshold:
                references.append(hypothesis)
                image_urls.append(line['result_url'])
    references,image_urls=no_duplicate(references,image_urls)
    return references,image_urls

def no_duplicate(references,image_urls):
    '''
    Deduplicate
    '''
    a={}
    for j,reference in enumerate(references):
        url=image_urls[j]
        a[reference]=url
    rs=[]
    us=[]
    for reference in a.keys():
        rs.append(reference)
        us.append(a[reference])
    return rs,us
    

def clip_embedding_retrieval(model, lines, user, query='cat', num=3):
    '''
    EBR based Retriever

    Args:
        model: CLIP model
        lines: User histories
        user: User id
        query: Current Query
        num: Number of retrievals results

    Returns:
        Dict including retrievals results:
            'user_id': user id
            'retrievals': retrieval results
            'raw_prompts': top-k relevant prompts
    '''
    sentences , image_urls = checklines(lines)
    sentences=list(set(sentences))
    retrievals=[]
    result=clip_similarity(model,query,sentences)
    kk=num*2 if len(sentences)>num*2 else len(sentences)
    scores, indices = result['score'].topk(kk)
    retrievals=[ [sentences[indices[i]],scores[i],image_urls[indices[i]]] for i in range(kk)]
    rt_sentences= [sentences[i] for i in indices]
    return {'user_id': user, 'retrievals': retrievals[:num], 'raw_prompts': rt_sentences}


#BM25 Model
#Code borrowed from https://github.com/Ricardokevins/Kevinpro-NLP-demo/blob/main/QuerySearch/query.py
class BM25_Model(object):
    def __init__(self, documents_list, k1=2, k2=1, b=0.5):
        self.documents_list = documents_list
        self.documents_number = len(documents_list)
        self.avg_documents_len = sum([len(document) for document in documents_list]) / self.documents_number
        self.f = []
        self.idf = {}
        self.k1 = k1
        self.k2 = k2
        self.b = b
        self.init()

    def init(self):
        df = {}
        for document in self.documents_list:
            temp = {}
            for word in document:
                temp[word] = temp.get(word, 0) + 1
            self.f.append(temp)
            for key in temp.keys():
                df[key] = df.get(key, 0) + 1
        for key, value in df.items():
            self.idf[key] = np.log((self.documents_number - value + 0.5) / (value + 0.5))

    def get_score(self, index, query):
        score = 0.0
        document_len = len(self.f[index])
        qf = Counter(query)
        for q in query:
            if q not in self.f[index]:
                continue
            score += self.idf[q] * (self.f[index][q] * (self.k1 + 1) / (
                    self.f[index][q] + self.k1 * (1 - self.b + self.b * document_len / self.avg_documents_len))) * (
                             qf[q] * (self.k2 + 1) / (qf[q] + self.k2))

        return score

    def get_documents_score(self, query):
        score_list = []
        for i in range(self.documents_number):
            score_list.append(self.get_score(i, query))
        return score_list

def bm25_retrieval(lines,user,query='cat',num=3):
    '''
    BM25 Retriever

    Args:
        lines: User histories
        user: User id
        query: Current Query
        num: Number of retrievals results

    Returns:
        Dict including retrievals results:
            'user_id': user id
            'retrievals': retrieval results
            'raw_prompts': top-k relevant prompts
    '''
    sentences, image_urls = checklines(lines)
    docs=[word_tokenize(doc.lower()) for doc in sentences]
    bm25=BM25_Model(docs)
    scores=bm25.get_documents_score(query)
    s_sentences=sorted(sentences,key=lambda x: scores[sentences.index(x)],reverse=True)
    s_image_urls = sorted(image_urls, key=lambda x: scores[image_urls.index(x)], reverse=True)
    sort_scores=sorted(scores,reverse=True)
    if num> len(sentences):
        num=len(sentences)
    retrievals=[[s_sentences[i],sort_scores[i],s_image_urls[i]] for i in range(num)]
    if num*2>len(sentences):
        ll=len(sentences)
    else:
        ll=num*2
    return {'user_id':user,'retrievals':retrievals,'raw_prompts':s_sentences[:ll]}


if __name__=='__main__':
    # import spacy
    # nlp=spacy.load('en_web_trf')
    pass