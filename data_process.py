# Code by zzjchen
# This code includes functions used for data processing
# The release PIP dataset is already processed
import time
import os
from utils import read_jsonl_data,write_jsonl_data
from apiuse import MY_ORG,MY_OPENAI_KEY,stable_call_api
import openai
import random
import argparse

def preprocess_line(line):
    '''
    Removing repetitive spaces and suffix
    '''
    if '  ' in line:
        line = line.replace('  ', ', ')
    if 'trending in artstation' in line:
        line = line.replace('trending in artstation', '')

    return line


def process_line(line, model_name='gpt-3.5-turbo', message_template='', t_sleep=2, max_tokens=1000):
    '''
    Abbreviating the prompt using ChatGPT
    '''
    prompt = line['prompt']
    prompt = preprocess_line(prompt)
    if prompt.count(' ') < 6:
        line['query'] = prompt
    else:
        messages = [
            {
                'role': 'user',
                'content': message_template.format(prompt)
            }
        ]
        short_prompt = stable_call_api(model_name=model_name, messages=messages, return_dict=False, t_sleep=t_sleep,
                                       max_tokens=max_tokens)
        line['query'] = short_prompt
    return line, prompt
def summarize_query(lines):
    '''
    Summarizing prompts in test samples to short prompts, using ChatGPT

    Args:
        lines: test sample

    Returns:
        test sample containing short prompt.
    '''
    new_lines=[]
    mes_template=[
        'Your task is to abbreviate a given text-to-image prompt. The abbreviated prompt should just include the primary object from the original prompt and basic attributes, ignoring trivial details and other descriptions. Generally speaking, an abbreviated prompt should be less than 10 words.',
        'Please abbreviate the following prompt:',
        'The original prompt: {}',
        'The abbreviated prompt (less than 10 words):'
    ]
    mes_template='\n'.join(mes_template)
    for line in lines:
        new_line,prompt=process_line(line,message_template=mes_template)
        if new_line['query']==prompt:
            new_line['good']=False
        else:
            new_line['good']=True
        new_lines.append(new_line)
    return new_lines



def split_a_user(lines):
    '''
    Split out 2 test samples from a user
    Args:
        lines: List of a users T2I histories

    Returns:
        train samples, test samples of the user
    '''
    prompts=[l['prompt'] for l in lines]
    prompts=list(set(prompts))
    cnt=[0]*len(prompts)
    for l in lines:
        id=prompts.index(l['prompt'])
        cnt[id]+=1
    s_prompts=sorted(prompts,key=lambda x: cnt[prompts.index(x)])
    s_cnt=sorted(cnt)
    threshold=s_cnt[2]
    rev_index=list(reversed(s_cnt)).index(threshold)
    test_prompts=s_prompts[:len(s_prompts)-rev_index]
    random.shuffle(test_prompts)
    test_prompts=test_prompts[:2]
    test_lines=[]
    for i in test_prompts:
        for l in lines:
            if l['prompt']==i:
                test_lines.append(l)
                break
    train_lines=[]
    for l in lines:
        if not l in test_lines:
            train_lines.append(l)
    return train_lines,test_lines


def process_a_user(lines):
    train_lines,test_lines=split_a_user(lines)
    test_lines=summarize_query(test_lines)
    return train_lines,test_lines




if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rawdir', type=str, default='tmp')

    args = parser.parse_args()
    os.makedirs('user_data')

    users=os.listdir(args.rawdir)
    for user in users:
        lines=read_jsonl_data(os.path.join('user_data',user))
        train,test=process_a_user(lines)
        res=['train_samples:']
        res.extend(train)
        res.append('test_samples:')
        res.extend(test)
        write_jsonl_data(res,os.path.join('user_data',user))





