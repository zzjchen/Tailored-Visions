# Code by zzjchen
# This code includes functions for ChatGPT API. Requires your OpenAI api key for usage.
# Login and visit https://platform.openai.com/account/api-keys to see your OpenAI api key
import openai
#Replace the following openai key, organization with your openai key or organization
#If you have special api_base for api.openai.com, replace My_Base with your correct api_base
MY_OPENAI_KEY=""
MY_ORG=""
MY_BASE='https://api.openai.com/v1/'
import time
def stable_call_api(model_name="gpt-3.5-turbo",messages=[{'role':'user','content':'hello'}],return_dict=False,max_tokens=500,t_sleep=0.2):
    '''
    A simple & stable wrapper for calling ChatGPT API, which keeps retrying after failure.
    Make sure you're able to connect the internet.

    Args:
        model_name: OpenAI model name. ChatGPT by default
        messages: Input message histories to ChatGPT
        return_dict: False if you just want to receive the string ChatGPT responds.
        max_tokens: Maximum of tokens ChatGPT is going to respond
        t_sleep: seconds until next retry in case of failure

    Returns:
        Chatgpt response message
    '''
    flg=True
    skip=t_sleep
    k=1
    while flg:
        try:
            message=call_api(model_name, messages, return_dict, max_tokens)
            flg = False
        except Exception as e:
            print('  ',e,'Retrying', k, 'sleep', skip)
            time.sleep(skip)
            skip+=t_sleep
            continue
    return message
def call_api(model_name="gpt-3.5-turbo",messages=[{'role':'user','content':'hello'}],return_dict=False,max_tokens=3500):
    '''
        A simple wrapper for calling ChatGPT API.

        Args:
            model_name: OpenAI model name. ChatGPT by default
            messages: Input message histories to ChatGPT
            return_dict: False if you just want to receive the string ChatGPT responds.
            max_tokens: Maximum of tokens ChatGPT is going to respond

        Returns:
            Chatgpt response message
    '''
    completion = openai.ChatCompletion.create(
      model=model_name,
      messages=messages,
      max_tokens = max_tokens
    )

    ans=completion.choices[0].message
    if return_dict:
        return ans
    else:
        return ans['content']

if __name__ == '__main__':
    openai.organization = MY_ORG
    openai.api_key = MY_OPENAI_KEY
    mes=[{'role': 'user',
          'content' :"hello"
          }]
    print('User:',mes[0]['content'])
    message=call_api(messages=mes,return_dict=True)
    print('ChatGPT:',message)
