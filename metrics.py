# Code by zzjchen
# This code includes functions for evaluation
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
import torch
import clip

@torch.no_grad()
def clip_score(model,preprocess,image,text,device='cuda'):
    '''
    Implementation of CLIPScore https://arxiv.org/abs/2104.0871 . 
    CLIPScore uses CLIP ViT-B/32 .
    Calculates CLIPScore(image,text) as the original article

    Args:
        model: CLIP model
        preprocess: CLIP image preprocess
        image: image
        text: text
        device: specify if not 'cuda'

    Returns:
        result of CLIPScore(image,text)
    '''
    image = preprocess(image).unsqueeze(0).to(device)
    text=clip.tokenize(text).to(device)
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    score=2.5*(image_features @ text_features.T)
    if score<0:
        score=0.0
    return score

@torch.no_grad()
def img_align(model,preprocess,ori_image,gen_image,device='cuda'):
    '''
    Implementation of Image-Align metric proposed, which calculates the similarity
    between ground truth image and generated image.
    Image-Align uses CLIP ViT-B/32 model

    Args:
        model: CLIP model
        preprocess: CLIP image preprocess
        ori_image: ground truth image
        gem_image: generated image
        device: specify if not 'cuda'
    '''
    ori_image = preprocess(ori_image).unsqueeze(0).to(device)
    gen_image = preprocess(gen_image).unsqueeze(0).to(device)
    ori_features = model.encode_image(ori_image)
    gen_features = model.encode_image(gen_image)
    gen_features /= gen_features.norm(dim=-1, keepdim=True)
    ori_features /= ori_features.norm(dim=-1, keepdim=True)
    score=2.5*(ori_features @ gen_features.T)
    if score<0:
        score=0.0
    return score


def calculate_bleu(references,hypothesis):
    '''
    Wrapper for BLEU Score calculation, used for removing very similar historical prompts.

    Args:
        references: List of reference text
        hypothesis: Current prompt text

    Returns:
        result of BLEU score
    '''
    refs=[ref.split() for ref in references]
    hyp=hypothesis.split()
    return sentence_bleu(refs,hyp)

if __name__=="__main__":
    pass
