# Code by zzjchen
# This code includes input example for prompt rewriting.
import random
import clip
import torch


# Instruction for prompt rewriting
Example_template = [
    "Prompt in text-to-image generation describes the detailed attributes of the object it plans to draw. User's preference in text-to-image generation is shown in history prompts.",
    "Given 3 history prompts, your task is to rewrite the current prompt so that it matches the user’s preference. The rewritten prompt should retain primary objects in the original prompt and conform to the user’s preference. Please avoid being too diffused and restrict your output within 70 words.",
]

# Template for ICL examples
History_template=[
    "Example {}:",
    "The history prompts are:",
    "1. {}",
    "2. {}",
    "3. {} \n",
    "The new query is: {}",
    "The rewritten prompt is :{}\n\n"
]

# 5 ICL examples we created
Example_1=[
    'Over exposure, sharp depth of field, cinematic lighting, bright, natural colors, real colors, iso noise Volumetric Lighting,\nBacklight, night, 3D, realistic, midjournal portal, a girl sitting on a park lounge chair with a delicate Eastern face, big eyes, long eyelashes, delicate lips, and a few ducklings swimming in the river in the distance. There are flowers and plants by the river, and a small pavilion nearby. HDR, the highest image quality, high,definition portrayal, sunset light, and a sense of atmosphere,trending in artstation',
    'A woman rides a white horse galloping on the green grassland, with long golden hair flying, wearing a dress on her head, beautiful green eyes, delete face, delete hands, wearing a red transparent coat Inside the gauge coat, she wears a white long skirt, and behind her fly a white eagle, with 8K image quality, HDR, UHD, lighting, the highest image quality, masterpiece, fine portal, exit CG, high light, ultra clear and detailed, with a sense of atmosphere, depth of field, bokeh, pristine lighting, photosensitive environment, trending on Artstation, 4k, 8k,CG_Render',
    'Woman, with golden long hair, wild flower wreath on her head, beautiful blue eyes, exquisite duck egg face, slightly longer face shape, exquisite mouth, pink lip glaze, wearing a red transparent gauze garment, wearing a white long skirt inside the gauze garment, exquisite necklace, silver hair accessories, high mountain waterfalls, faint castles, sunset on the horizon, birds, light, dreamy and charming, looking at the waterfall, back to the camera, full body, highest picture quality, masterpiece, fine portrayal, exquisite CG, Natural light, ultra clear and detailed,CG_Render',
    'A woman stands in front of a castle',
    'A woman stands in front of a high mountain castle, with long brown hair, beautiful blue eyes, exquisite duck egg face, slightly longer face shape, exquisite mouth, pink lip glaze, wearing a transparent red gauze garment. Inside the gauze garment, she is wearing a white long skirt, exquisite necklace, silver hair accessories, flowing water waterfall, sunset on the horizon, beautiful birds, and light. The woman looks at the waterfall, smiling and not smiling, with her back to the camera, her whole body, the highest picture quality, masterpiece, fine portrayal, and exquisite CG, Natural light, ultra clear and detailed,CG_Render,trending in artstation'
]
Example_2=[
    'Close,up of a boy with blue eyes and white hair, surrounded by a golden halo and an angel halo, Impressionism, Claude Monet, Alfred Sisley, Vincent Willem van Gogh',
    'Back view, a boy carrying a backpack home, with a white Samoye at his feet. The sky is blue, and on the green grass, there are many scattered pink flowers,UnrealEngine, CG_Render',
    'Gentle white Samoye with pink and blue flowers and azure sky dragon sky next to it, Gamescene, trending in artstation ',
    'Little Samoye followed behind the hostess',
    'Little Samoye followed behind the hostess, surrounded by pink flowers and azure sky, Gamescene, trending in artstation'
]
Example_3=[
    'Best Quality, Masterpiece photo of a girl, Single Person, Perfect Eyes, Acquire Face, Acquire Skin, Black Hair, Long Hair, Necklace, Looking at Camera, City, Street, Night, Hotel, Skyscraper, Skyline, Top Shoulder, Bottom Split Short Skirt, Black Socks, A Pair of Red High Heels, 4k Ultra Clear, Highly detailed, pro Professional digital painting, Unreal Engine 5, Photorealism, HD quality, 8k resolution, cinema 4d, 3D, cinematic, professional photography, art by art and greg rutkowski and alphonse mucha and finish and WLOP,Photography',
    'Photography exquisitely portrays realistic characters in the style of Pino daeni, a female college student fashion masterpiece, a classic masterpiece, a close,up of a beautiful girl, black hair, side bun, red dress, happy, long hair fluttering, full moon, dark night, glowing red fireflies, amidst plum blossoms, forest in the distance, colorful and colorful oil painting techniques, the best composition,Photography, realistic.',
    'A beautiful photo of a European girl, exquisite and complex skirt, colorful clothing, mature and beautiful, with clear and moving eyes, sitting outdoors, (masterpiece), best quality, midjournal portrait, masterpiece, close,up, by Paul Hedley.',
    'A beautiful girl',
    'A delicate and beautiful girl with a pure temperament, red lips, black hair, straight hair, wearing a black off shoulder top, red split mini skirt, high heels, modern urban clothing. Standing on the evening street, with trees on both sides of the street and leaves falling all over the street, the whole body is photographed, realistic, 8k, The best quality, masterpiece, highlights, beautiful.'
]
Example_4=[
    'Black hair, handsome and cute esports man, best quality, full details, realistic photography',
    '(masterpiece), best quality, close up of a boy with black eyes and hair, smiling face, cat ears, CG_rendering, 8k uhd, trending in artstation',
    'A handsome and cute teenager with yellow hair and white cloth clothes and a sword, 8k, high quality, trending in artstation.',
    'Young man with black hair',
    '(Masterpiece), best quality, handsome and cute young man with black hair and white cloth clothes holding a sword, exquisite details, trending in artstation.'
]
Example_5=[
    "The character is a young girl. The background is a lush forest, surrounded by flowers and trees, the setting sun, dusk, and halo. The painting style is Hayao Miyazaki's style. Emotions are surprises, happiness, doubts, and looking,Cinematic",
    "Ghibli, a work by Hayao Miyazaki, features gouache colors, high saturation, and soft lines. The blue sky character is a young girl. The posture is wearing glasses, looking very gentle, with eyes focused on the front, and holding a pile of books in one's arms. Wearing simple, expressionless, with exquisite facial features and tied high ponytail. The background is the library,GhibliStudio",
    "The character is a Japanese romantic girl. Wearing a jk outfit, black stockings, and wearing eyes. Sunny and outgoing personality, yet full of scheming. The background is the football field on campus,undefined",
    'A little girl next door.',
    "The character is a little girl next door. The action is to ride a bicycle leisurely. Emotionally reserved, obedient, happy. Hayao Miyazaki's Painting Style. The scene is the setting sun, dusk, the seaside,Cinematic"
]

Examples=[Example_1,Example_2,Example_3,Example_4,Example_5]

@torch.no_grad()
def rank_examples(query,examples):
    '''
    Ranks examples according to similarity with current query

    Args:
        query: current prompt
        examples: unsorted ICL examples

    Returns:
        examples: sorted ICL examples
    '''
    device='cuda' if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)
    query=clip.tokenize(query).to(device)
    texts=[e[-2] for e in examples]
    texts=clip.tokenize(texts).to(device)
    query_f=model.encode_text(query)
    texts_f=model.encode_text(texts)
    scores=texts_f @ query_f.T
    scores=scores.cpu().numpy()
    sorted_ex = sorted(examples, key=lambda x: scores[examples.index(x)],reverse=True)
    return sorted_ex


def get_ICL_example(num=3,k=1,query=''):
    '''
    Get ChatGPT input for context-dependent prompt rewrite given a query

    Args:
        num: number of retrieval histories.
        k: number of ICL examples
        query: current query

    Returns:
        String containing input for ChatGPT
    '''
    Es=rank_examples(query,Examples)
    mes_template='\n'.join(Example_template)
    for i in range(k):
        mes_template+=('\n'+'\n'.join(History_template).format(i+1,*Es[i]))
    mes_template+=('\n'+'Question:\n')
    if num==1:
        a = [
            "The history prompts are:",
            "1. {}\n",
            "The new query is: {}",
            "The rewritten prompt (one sentence less than 70 words) is :"
        ]
    elif num==3:
        a = [
            "The history prompts are:",
            "1. {}",
            "2. {}",
            "3. {} \n",
            "The new query is: {}",
            "The rewritten prompt (one sentence less than 70 words) is :"
        ]
    elif num==5:
        a = [
            "The history prompts are:",
            "1. {}",
            "2. {}",
            "3. {}",
            "4. {}",
            "5. {}\n",
            "The new query is: {}",
            "The rewritten prompt (one sentence less than 70 words) is :"
        ]
    else:
        a=[
            "The history prompts are:",
            "1. {}",
            "2. {}",
            "3. {}",
            "4. {}",
            "5. {}",
            "6. {}",
            "7. {} \n",
            "The new query is: {}",
            "The rewritten prompt (one sentence less than 70 words) is :"
        ]
    astr='\n'.join(a)
    return mes_template+astr

def get_ZS_example(num=3):
    '''
    Get ChatGPT input for context-independent prompt rewrite given a query

    Args:
        num: number of retrieval histories

    Returns:
        String containing input for ChatGPT
    '''
    mes_template='\n'.join(Example_template)
    if num==1:
        a = [
            "The history prompts are:",
            "1. {}\n",
            "The new query is: {}",
            "The rewritten prompt (one sentence less than 70 words) is :"
        ]
    elif num==3:
        a = [
            "The history prompts are:",
            "1. {}",
            "2. {}",
            "3. {} \n",
            "The new query is: {}",
            "The rewritten prompt (one sentence less than 70 words) is :"
        ]
    elif num==5:
        a = [
            "The history prompts are:",
            "1. {}",
            "2. {}",
            "3. {}",
            "4. {}",
            "5. {}\n",
            "The new query is: {}",
            "The rewritten prompt (one sentence less than 70 words) is :"
        ]
    else:
        a=[
            "The history prompts are:",
            "1. {}",
            "2. {}",
            "3. {}",
            "4. {}",
            "5. {}",
            "6. {}",
            "7. {} \n",
            "The new query is: {}",
            "The rewritten prompt (one sentence less than 70 words) is :"
        ]
    astr='\n'.join(a)
    return mes_template+astr

def get_example(num=3,k=0,query=''):
    '''
    Wrapper for context-dependent & context-independent prompt rewrite.
    Context-independent prompt rewrite only uses one argument: num

    Args:
        num: number of retrieval histories.
        k: number of ICL examples
        query: current query

    Returns:
        String containing input for ChatGPT
    '''
    if k==0:
        return get_ZS_example(num)
    else:
        return get_ICL_example(num,k,query)

if __name__ == '__main__':
    print(get_example(3))













