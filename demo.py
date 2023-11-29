# Code by zzjchen
# This code is a demo of our method. To run our method on full PIP dataset, please refer to main.py
import os
from language import clip_embedding_retrieval,bm25_retrieval
from apiuse import stable_call_api,MY_OPENAI_KEY,MY_BASE
import openai
import torch
import clip
from utils import read_jsonl_data
from prompts import get_ICL_example
from SD import text2img
import argparse

#create prompt
def create_chat_prompt(history_prompts,query,prompt_template):
    if len(history_prompts)+1==prompt_template.count('{}'):
        prompt=prompt_template.format(*history_prompts,query)
    else:
        history_prompts.extend(["","",""])
        history_prompts=history_prompts[:prompt_template.count('{}')-1]
        prompt=prompt_template.format(*history_prompts,query)
    return prompt

#prompt_rewrite
def prompt_rewriting(history_prompts,query,prompt_template):
    chatgpt_prompt= create_chat_prompt(history_prompts, query, prompt_template)
    messages=[{'role':'user','content':chatgpt_prompt}]

    chatgpt_response=stable_call_api(messages=messages, return_dict=True)
    return {
        'user': chatgpt_prompt,
        'new_prompt':chatgpt_response['content']
    }


#clip_embed_retrieve+rewrite
@torch.no_grad()
def prompt_rewrite(clip_model,lines,user,query='cat',num=3,prompt_template=''):
    query=query.replace('\n','')
    query=query.replace('\r','')
    #query= re.sub(r'["\'?]', '', query)
    retrieval_result=clip_embedding_retrieval(clip_model, lines, user, query, num)
    sentences=retrieval_result['raw_prompts']
    retrieval_history=retrieval_result['retrievals']
    history_prompts=[i[0] for i in retrieval_history]
    rewrite_result=prompt_rewriting(history_prompts,query,prompt_template)
    return rewrite_result['new_prompt'].replace('\n',' ')
#check and translate
def is_contains_chinese(strs):
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            return True
    return False

def check_and_translate(strs):
    if is_contains_chinese(strs):
        chatgpt_prompt="translate the following sentence into English (answer in full English sentence):\n"+strs
        messages=[{'role':'user','content':chatgpt_prompt}]
        strs=stable_call_api(messages=messages, return_dict=True)['content']
    return strs
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default='cat')
    parser.add_argument('--t2i', action="store_true",default=True)
    args = parser.parse_args()
    openai.api_key = MY_OPENAI_KEY
    openai.api_base=MY_BASE
    data_folder='user_data'
    user_id='87403'
    query=args.prompt
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    # clip
    clip_model, _ = clip.load("ViT-L/14", device=device)
    clip_model.eval()
    user_path='demo_user.jsonl'
    if not os.path.isfile(user_path):
        print(user_id,'is not a valid user.')
    else:
        lines = read_jsonl_data(user_path)
        if lines[0]=='Preferences:':
            lines=lines[2:-2]
        if len(lines)<8:
            print(user_id,'does not have enough T2I histories.')
        else:
            prompt_template = get_ICL_example(3, 1, query)
            new_prompt = prompt_rewrite(clip_model, lines, user_id, query=query, num=3, prompt_template=prompt_template)
            print('Original Prompt:',query)
            print('Rewriten Prompt:',new_prompt)
            if args.t2i:
                model_id = "runwayml/stable-diffusion-v1-5"
                pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
                pipe = pipe.to(device)
                text2img(pipe,query,save=True,save_path="ori_prompt.png")
                text2img(pipe,new_prompt,save=True,save_path="personalized_prompt.png")
