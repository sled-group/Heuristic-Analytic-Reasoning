from PIL import Image
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import openai
from pprint import pprint
from io import BytesIO
import numpy as np
from typing import Optional, Union, List, Dict
from functools import partial
import time
from llama_cpp import Llama
import pickle
from fastchat.conversation import Conversation, SeparatorStyle
import sys
from visualization import get_output_line_sep, calc_attn

from utils import imload

LM_BACKBONE_ENGINES = {'gpt4': 'gpt-4', 'chatgpt': 'gpt-35-turbo'}

# (per token)
API_COSTS = {
    "gpt3": (0.00002, 0.00002),
    "chatgpt": (0.000002, 0.000002),
    "gpt4": (0.00006, 0.00012)
}

VICUNA13B_PATH = "/nfs/turbo/coe-chaijy/pre-trained-weights/vicuna/vicuna-13b-delta-v1.1"
ALPACA13B_PATH = "./llama.cpp/models/ggml-alpaca-13b-q4.bin"

# Prompt GPT-3
def prompt_gpt3(api_key: str, prompt: str, n: int=1, temperature: float=0.0, max_tokens: int=100, endpoint_name: str='lg-multimodal-lms-openai', engine: str="lg-multimodal-lms-gpt3") -> List[str]:
    openai.api_type = "azure"
    openai.api_base = f"https://{endpoint_name}.openai.azure.com/"
    # openai.api_version = "2022-06-01-preview"
    openai.api_version = "2022-12-01"
    openai.api_key = api_key

    results = []
    results_to_generate = n
    while results_to_generate > 0: # OpenAI APIs have a limit of 10 generations per time
        this_n = results_to_generate if results_to_generate < 10 else 10
        try:
            response = openai.Completion.create(engine=engine,
                                                prompt=prompt,
                                                temperature=temperature,
                                                n=this_n,
                                                max_tokens=max_tokens,
                                                frequency_penalty=0,
                                                presence_penalty=0,
                                                stop=None)
        except openai.error.RateLimitError:
            time.sleep(1)
            continue  
        except openai.error.AuthenticationError:
            raise
        except Exception as e:
            print("Exception:", e)
            return []          
        results += [choice['text'] for choice in response['choices']]
        results_to_generate -= this_n
    assert len(results) == n
    return results

# Prompt Codex
def prompt_codex(api_key: str, prompt: str, n: int=1, temperature: float=0.0, max_tokens: int=100, endpoint_name: str='lg-multimodal-lms-openai', engine: str="lg-multimodal-lms-gpt3") -> List[str]:
    openai.api_type = "azure"
    openai.api_base = f"https://{endpoint_name}.openai.azure.com/"
    # openai.api_version = "2022-6-01-preview"
    openai.api_version = "2022-12-01"
    openai.api_key = api_key

    results = []
    results_to_generate = n
    while results_to_generate > 0: # OpenAI APIs have a limit of 10 generations per time
        this_n = results_to_generate if results_to_generate < 10 else 10
        try:
            response = openai.Completion.create(engine=engine,
                                                prompt=prompt,
                                                temperature=temperature,
                                                n=this_n,
                                                max_tokens=max_tokens,
                                                frequency_penalty=0,
                                                presence_penalty=0,
                                                stop=None)
        except openai.error.RateLimitError:
            time.sleep(1)
            continue                        
        results += [choice['text'] for choice in response['choices']]
        results_to_generate -= this_n
    assert len(results) == n
    return results

# Prompt DALL-E 2 to generate an image from text, or text and an image
def prompt_dalle2(api_key: str, prompt: str, n: int=1, input_image: Image=None, input_mask: Image=Image.new('RGBA', (256, 256), (0, 0, 0, 0))):
    openai.api_type = "open_ai"
    openai.api_base = "https://api.openai.com/v1"
    openai.api_version = None
    openai.api_key = api_key

    if input_image is None:
        results = []
        results_to_generate = n
        while results_to_generate > 0: # OpenAI APIs have a limit of 10 generations per time
            this_n = results_to_generate if results_to_generate < 10 else 10
            response = openai.Image.create(
                prompt=prompt,
                n=this_n,
                size="256x256"
            )
            # In case of connection issues here
            for _ in range (5):
                try:
                    results += [imload(data['url']) for data in response['data']]
                    break
                except:
                    continue
            results_to_generate -= this_n
    else:
        results = []
        results_to_generate = n
        while results_to_generate > 0: # OpenAI APIs have a limit of 10 generations per time
            this_n = results_to_generate if results_to_generate < 10 else 10

            # Get bytes of input image and mask
            input_image = input_image.resize((256, 256))
            input_image.save('temp_image.png')
            # byte_stream = BytesIO()
            # input_image.resize((256, 256)).save(byte_stream, format="PNG")
            # input_image = byte_stream.getvalue()

            # Convert binary mask array to PIL image
            input_mask = input_mask.resize((256, 256))
            input_mask.save('temp_mask.png')
            # byte_stream = BytesIO()
            # input_mask.resize((256, 256)).save(byte_stream, format="PNG") # shouldn't need to be resized but just in case
            # input_mask = byte_stream.getvalue()

            response = openai.Image.create_edit(
                image=open('temp_image.png', 'rb'),
                mask=open('temp_mask.png', 'rb'),
                prompt=prompt,
                n=this_n,
                size="256x256"
            )

            # In case of connection issues here
            for _ in range (5):
                try:
                    pprint(response)
                    results += [imload(data['url']) for data in response['data']]
                    break
                except:
                    continue
            results_to_generate -= this_n

    assert len(results) == n
    return results



def get_chat_message(role: str, message: str, model_name='gpt4'):
    assert role in ['system', 'assistant', 'user']
    if model_name in ['gpt4', 'chatgpt']:
        return {'role': role,
                'content': message}
    elif model_name in ['vicuna13b']:
        return [role, message]

# Prompt ChatGPT or GPT-4
def prompt_chat_gpt(api_key: str, messages, n: int=1, temperature: float=0.0, max_tokens: int=100, engine: str="gpt-4") -> List[str]:
    if engine == 'gpt-4':
        # Use OpenAI's GPT-4 API
        openai.api_type = "azure"
        openai.api_base = "https://lg-multimodal-lms-openai.openai.azure.com/"
        openai.api_version = "2023-03-15-preview"
        engine = "lg-multimodal-lms-gpt4"
    elif engine == 'gpt-35-turbo':
        # But Azure's ChatGPT API
        openai.api_type = "azure"
        openai.api_base = f"https://lg-multimodal-lms-openai.openai.azure.com/"        
        openai.api_version = "2023-03-15-preview"
        engine = "lg-multimodal-lms-chatgpt"
    else:
        raise ValueError("Model %s not supported with chat prompts." % engine)
    openai.api_key = api_key

    results = []
    results_to_generate = n
    while results_to_generate > 0: # OpenAI APIs have a limit of 10 generations per time
        this_n = results_to_generate if results_to_generate < 10 else 10
        try:
            response = openai.ChatCompletion.create(
                engine=engine,
                messages=messages,
                temperature=temperature,
                n=this_n,
                max_tokens=max_tokens
            )
        except openai.error.RateLimitError:
            time.sleep(1)
            continue
        except openai.error.AuthenticationError:
            raise
        except:
            return []
        results += [choice['message']['content'] if 'content' in choice['message'] else "" for choice in response['choices']]
        results_to_generate -= this_n
    assert len(results) == n
    return results

def prompt_alpaca(prompt, max_tokens: int=100, engine="alpaca13b"):
    llm = None
    if engine == "alpaca13b":
        llm = Llama(model_path=ALPACA13B_PATH)
    output = llm(prompt, max_tokens=max_tokens)
    return output

def prompt_hf(model, tokenizer, prompt, temperature: float=0.0, max_tokens=128, output_attn=False):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = inputs.to('cuda:0')
    generate_ids = model.generate(inputs.input_ids, max_new_tokens=max_tokens)
    attn, line_sep, tokens = None, None, None
    if output_attn:
        attn = generate_ids["attentions"]
        generate_ids = generate_ids["sequences"]
        tokens = tokenizer.convert_ids_to_tokens(generate_ids[0]) 
        line_sep = get_output_line_sep(tokens, inputs.input_ids.shape[1])
    return tokenizer.batch_decode(
        generate_ids[:, inputs.input_ids.shape[1]:], 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False, 
        do_sample=False if temperature == 0.0 else True, 
        temperature=temperature
    )[0], attn, line_sep, tokens, inputs.input_ids.shape[1]

def prompt_fastchat(model, tokenizer, messages, temperature: float=0.0, max_tokens=128):
    messages = Conversation(
        name="structured-cot",
        system=messages[0][1],
        roles=['system', 'user'],
        messages=messages[1:],
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep='\n\n',
    )
    input_ids = tokenizer([messages.get_prompt()]).input_ids
    output_ids = model.generate(
        torch.as_tensor(input_ids).cuda(),
        do_sample=True if temperature > 0.0 else False,
        temperature=temperature,
        max_new_tokens=max_tokens,
    )
    if model.config.is_encoder_decoder:
        output_ids = output_ids[0]
    else:
        output_ids = output_ids[0][len(input_ids[0]) :]
    outputs = tokenizer.decode(
        output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
    )            
    return outputs

def prompt_gpt3_with_caching(prompt, args, lm_cache, cache_key, max_tokens=128, token_counts=None):
    # print("# tokens:", len(prompt.split())/3*4)
    if not args.skip_prompting:
        if cache_key not in lm_cache:
            if getattr(args, "cache_only", None) and args.cache_only:
                generated_text = "Story A"
            else:
                generated_text = prompt_gpt3(args.api_key,
                                            prompt,
                                            temperature=0,
                                            max_tokens=max_tokens)
                if len(generated_text) > 0:
                    generated_text = generated_text[0]
                    lm_cache[cache_key] = generated_text
                    pickle.dump(lm_cache, open(args.cache_fname, 'wb'))
                else:
                    return "Story A"
        else:
            generated_text = lm_cache[cache_key]
        return generated_text
    else:
        if cache_key not in lm_cache:
            # Just estimate by words (not perfect)
            token_counts['prompt'] += len(prompt.split())
            token_counts['gen'] += max_tokens
        pprint(prompt)
        return "Story A"
    
def prompt_llama_with_caching(model, tokenizer, prompt, args, lm_cache, cache_key, max_tokens=128, output_attn=False, separations=None):
    if not args.skip_prompting:
        if (cache_key not in lm_cache 
        or (output_attn and cache_key + "_attn" not in lm_cache)):
            # print("prompt:", prompt)
            generation = prompt_hf(
                model, 
                tokenizer,
                prompt,
                max_tokens=max_tokens,
                output_attn=output_attn,
            )
            if output_attn:
                generated_text, attn, token_sep, tokens, prompt_len = generation
            else:
                generated_text, _, _, _, _ = generation
            # print("generated_text:", generated_text)
            if len(generated_text) > 0:
                lm_cache[cache_key] = generated_text
                if output_attn:
                    attn_layers = [i for i in range(model.config.num_hidden_layers)]
                    story_separation = separations[0]
                    layer_attn = []
                    for layer in attn_layers:
                        line_attn = []
                        for i in range(len(token_sep)-1):
                            token_range = [token_sep[i] - prompt_len, token_sep[i+1] - prompt_len]
                            full_attn = calc_attn(
                                    attn,
                                    token_range,
                                    [layer],
                                    tokens,
                                    ignore_special_tokens=True,
                                    separations=story_separation,
                                    separation_average=True,
                            )
                            story_attn = []
                            for sep in story_separation:
                                start = sep[0]
                                sentence_attn = []
                                for end in sep[1:]:
                                    sentence_attn.append(np.sum(full_attn[start:end]).tolist())
                                    start=end
                                story_attn.append(sentence_attn)
                            line_attn.append(story_attn)
                        layer_attn.append(line_attn)
                    lm_cache[cache_key + "_attn"] = layer_attn
                pickle.dump(lm_cache, open(args.cache_fname, 'wb'))
            else:
                return "Story A"
        else:
            generated_text = lm_cache[cache_key]
        return generated_text
    else:
        pprint(prompt)
        return "Story A"
    
def prompt_chat_gpt_with_caching(messages, args, lm_cache, cache_key, max_tokens=128, token_counts=None):
    if not args.skip_prompting:
        if cache_key not in lm_cache:
            if args.api_key == None:
                # raise Exception("No ChatGPT API key given.")
                print("No ChatGPT API key given.") # switch to this when evaluating cached file
            generated_text = prompt_chat_gpt(
                args.api_key,
                messages,
                max_tokens=max_tokens,
                engine=LM_BACKBONE_ENGINES[args.lm_backbone],
            )
            if len(generated_text) > 0:
                generated_text = generated_text[0]
                lm_cache[cache_key] = generated_text
                pickle.dump(lm_cache, open(args.cache_fname, 'wb'))
            else:
                return "Story A"
        else:
            generated_text = lm_cache[cache_key]
        return generated_text
    else:
        if cache_key not in lm_cache:
            # Just estimate by words (not perfect)
            token_counts['prompt'] += len(" ".join([message['content'] for message in messages]).split())
            token_counts['gen'] += max_tokens
        pprint(messages)
        return "Story A"

def prompt_alpaca_with_caching(prompt, args, lm_cache, cache_key, max_tokens=128):
    if not args.skip_prompting:
        if cache_key not in lm_cache:
            generated_text = prompt_alpaca(
                prompt,
                max_tokens=max_tokens,
                engine=LM_BACKBONE_ENGINES[args.lm_backbone],
            )
            if len(generated_text) > 0:
                generated_text = generated_text[0]
                lm_cache[cache_key] = generated_text
                pickle.dump(lm_cache, open(args.cache_fname, 'wb'))
            else:
                return "Story A"
        else:
            generated_text = lm_cache[cache_key]
        return generated_text
    else:
        pprint(prompt)
        return "Story A"
    
def prompt_fastchat_with_caching(model, tokenizer, messages, args, lm_cache, cache_key, max_tokens=128):
    if not args.skip_prompting:
        if cache_key not in lm_cache:
            generated_text = prompt_fastchat(
                model,
                tokenizer,
                messages,
                max_tokens=max_tokens,
            )
            if len(generated_text) > 0:
                lm_cache[cache_key] = generated_text
                pickle.dump(lm_cache, open(args.cache_fname, 'wb'))
            else:
                return "Story A"
        else:
            generated_text = lm_cache[cache_key]
        return generated_text
    else:
        pprint(messages)
        return "Story A"    
    
def get_65b_device_mapping(config, num_devices, num_layers_per_device):
    num_layers = config.num_hidden_layers
    assert(num_layers_per_device * (num_devices-1) >= num_layers)
    layer_map = {i: i//num_layers_per_device + 1 for i in range(num_layers)}
    last_device = num_devices - 1
    device_map = {
        'lm_head': last_device,
        'model.embed_tokens': last_device,
        'model.norm': last_device,
    }
    for i in range(num_layers):
        device_map[f"model.layers.{i}"] = layer_map[i]
    return device_map