# Code by zzjchen
# This code includes main code for Personalized Prompt Rewriting in Tailored Vision.
# This code requires GPU for T2I generation, turning 't2i' flag in '__call__' function to False to skip T2I generation
import os
from language import clip_embedding_retrieval,bm25_retrieval
from apiuse import MY_OPENAI_KEY,MY_ORG,stable_call_api
import openai
from SD import text2img
import torch
import clip
from diffusers import StableDiffusionPipeline
from utils import read_jsonl_data,write_jsonl_data
import time
import argparse
from prompts import get_example
import func_timeout
from download import wget_download

class Full_Pipeline:
    def __init__(self, clip_model, SD_pipe,retrieval='bm25',out_folder='image_result'):
        '''
        Full pipeline of personalized prompt rewriting in Tailored Vision includes a Retriever, a Rewriter and a T2I Generator
        We included:
            2 types of Retriever: BM25 & EBR
            2 types of prompt for Rewriter (ChatGPT): context-dependent & context-independent
            Stable Diffusion v1-5 as T2I Generator
        After initializing models, a folder will be created under out_folder for saving experiment results

        Args:
            clip_model: CLIP model used in EBR Retriever, ignored for BM25 Retriever
            SD_pipe: StableDiffusionPipeline for T2I Generator
            retrieval: 'bm25' or 'ebr', type of Retriever
            out_folder: Path where a folder will be created to save current experiment results
        '''
        self.retrieval=retrieval
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.retrieval=='bm25':
            self.clip_model=None
        else:
            self.clip_model=clip_model
        self.SD_pipe = SD_pipe
        self.SD_pipe.to(self.device)
        openai.organization = MY_ORG
        openai.api_key = MY_OPENAI_KEY
        openai.api_base = MY_BASE
        self.today=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
        self.out_folder=out_folder
        if not os.path.exists(os.path.join(out_folder,self.today)):
            os.mkdir(os.path.join(out_folder,self.today))

    def __call__(self,lines,user,query='cat',num=3,prompt_template='',t2i=False):
        '''
        Inferencing personalized prompt rewriting given a user history and a query

        Args:
            lines: List of personalized T2I histories from a user
            user: Current user id
            query: Current Query
            num: Number of Histories to be retrieved
            prompt_template: ChatGPT Input Template
            t2i: Bool indicating whether to perform T2I generation, or simply perform prompt rewriting

        Returns:
            Dict containing personalized prompt rewriting results:
                'user_id': user id
                'query': current query
                'retrieval_history': retrieval results
                'new_prompt': personalized prompt rewriting result
                'image_path': folder where the generated image and other results will be saved
        '''
        if self.retrieval=='full':
            retrieval_result = clip_embedding_retrieval(self.clip_model, lines, user, query, num)
        else:
            retrieval_result=bm25_retrieval(lines,user,query,num)
        sentences=retrieval_result['raw_prompts']
        root=self.create_folder(user,query)
        if t2i:
            ori=text2img(self.SD_pipe,query,1,save=True,save_path=os.path.join(root,'ori.png'),guidance_scale=7.0,num_inference_steps=50)[0]
        retrieval_history=retrieval_result['retrievals']
        history_prompts=[i[0] for i in retrieval_history]
        rewrite_result=self.prompt_rewriting(history_prompts,query,prompt_template)
        SD_prompt=rewrite_result['new_prompt']
        if t2i:
            gen=text2img(self.SD_pipe,SD_prompt,1,save=True,save_path=os.path.join(root,'generated.png'),guidance_scale=7.0,num_inference_steps=50)[0]
        if t2i:
            return {
                'user_id':user,
                'query':query,
                "retrieval_history":retrieval_history,
                "new_prompt":SD_prompt,
                "images_path":root,
            }
        else:
            return {
                'user_id': user,
                'query': query,
                "retrieval_history": retrieval_history,
                "new_prompt": SD_prompt,
                "images_path": root,
            }

    def create_folder(self,user_id,query):

        t=0
        query=query.split(' ')[0]
        while os.path.exists(os.path.join(self.out_folder,self.today,user_id+'-'+query+'-'+str(t))):
            t+=1
        url=os.path.join(self.out_folder,self.today,user_id+'-'+query+'-'+str(t))
        os.makedirs(url)
        return url


    def prompt_rewriting(self,history_prompts,query,prompt_template):
        chatgpt_prompt= self.create_chat_prompt(history_prompts, query, prompt_template)
        messages=[{'role':'user','content':chatgpt_prompt}]

        chatgpt_response=stable_call_api(messages=messages, return_dict=True)
        return {
            'user': chatgpt_prompt,
            'new_prompt':chatgpt_response['content']
        }
    def create_chat_prompt(self,history_prompts,query,prompt_template):
        assert len(history_prompts)+1==prompt_template.count('{}')
        prompt=prompt_template.format(*history_prompts,query)
        return prompt
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrieval', type=str, default='ebr')
    parser.add_argument('--num_retrieval', type=int,default=3)
    parser.add_argument('--rewrite_method', type=str, default='naive')
    parser.add_argument('--ICL_shot',type=int,default=0)
    parser.add_argument('--t2i',action='store_true')
    args = parser.parse_args()

    # Building models
    if args.retrieval=='bm25':
        clip_model,=None
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, _ = clip.load("ViT-L/14", device=device)
    if args.t2i==True:
        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    else:
        pipe=None
    FP=Full_Pipeline(clip_model,pipe,retrieval=args.retrieval)
    if args.rewrite_method=='ICL':
        mes_template=get_example(args.num_retrieval,args.ICL_shot)
    else:
        mes_template=get_example(args.num_retrieval,0)

    # Reading all users in PIP Dataset
    urls=os.listdir('user_data')

    bad_samples=[]
    # Performs personalized prompt rewriting for every user
    for i in range(len(urls)):
        o_url = urls[i]
        print('*****User:',o_url.split('.')[0],'*****')
        url = os.path.join('user_data', o_url)
        lines=read_jsonl_data(url)
        prompts=lines[1:-3]
        queries=lines[-2:]
        user = url.split('/')[-1][:-6].split('\\')[-1]
        for queryl in queries:
            try:
                query = queryl['query']
                target = queryl['prompt']
                print('    Query：', query)
                time.sleep(0.5)
                result = FP(prompts, user, query, num=args.num_retrieval, prompt_template=mes_template,t2i=args.t2i)
                result['retrieval_history'] = [i[:-1] for i in result['retrieval_history']]
                root = result['images_path']
                with open(os.path.join(root, 'result.txt'), 'w', encoding='utf-8') as f:
                    lines = []
                    def prepare(key):
                        '''
                        Preparing experiment results before saving in txt file
                        '''
                        value = result[key]
                        line = ''
                        if type(value) == list:
                            line += (key + ':\n')
                            for a in value:
                                line += ('\t' + str(a) + '\n')
                        else:
                            line += (key + ': ' + str(value) + '\n')
                        return line + '\n'

                    for key in result.keys():
                        lines.append(prepare(key))
                    f.writelines(lines)
            except Exception as e:
                print('Failed due to',str(e))
                bad_samples.append((o_url,queries.index(queryl),str(e)))

    print('Recording errors')
    write_jsonl_data(bad_samples, os.path.join('image_result',FP.today,'error.jsonl'))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
