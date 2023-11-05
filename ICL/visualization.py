from typing import List, Tuple
import torch

SPECIAL_TOKENS = {
    '<s>':'',
    '<0x0A>':'\\',
}

def convert_special_tokens(word_list, special_tokens: dict):
	new_word_list = []
	for word in word_list:
		for special_token in special_tokens:
			if special_token in word:
				word = word.replace(special_token, special_tokens[special_token])
		new_word_list.append(word)
	return new_word_list


def calc_token_idx(token_idx, prompt_len): # token list w/ prompt -> token list w/o prompt
    return token_idx-prompt_len


def get_output_line_sep(tokens, prompt_len, sep_tokens=["<0x0A>"]):
    output_tokens=tokens[prompt_len:]
    line_sep=[prompt_len]
    for idx, token in enumerate(output_tokens):
        if token in sep_tokens:
            line_sep.append(idx+prompt_len)
    return line_sep


def separation_generator(demo, story_prompt, tokenizer): # note that the final prompt should be demo + story_prompt, add '\n' in advance
    separations = []
    sep = tokenizer(demo, return_tensors='pt').input_ids.shape[-1]
    separations.append(sep)
    lines = story_prompt.split("\n")
    lines = [line+"\n" for line in lines if line != '']
    for idx, line in enumerate(lines):
         if idx == 0:
             continue
         if "Story" in line:
              story_len = idx
    # story_len = len(lines)//2
    for idx, line in enumerate(lines):
        sep += tokenizer(line, return_tensors='pt').input_ids.shape[-1] - 1
        if idx != 0 and idx != story_len: # if line start with numbers, an extra token will be added, need to subtract to keep track of lines
            sep -= 1
        separations.append(sep)
    # remove "Story A" and "Story B"
    story_a_sentences = separations[1:story_len+1]
    story_b_sentences = separations[story_len+1:]
    separations = [
        [story_a_sentences, story_b_sentences],
        [[story_a_sentences[0], story_a_sentences[-1]], [story_b_sentences[0], story_b_sentences[-1]]],
    ]
    return separations


def calc_attn(
        attentions: Tuple[Tuple[torch.Tensor]],
        token_ids_to_average: List[int],
        attn_layers: List[int], 
        tokens, 
        ignore_special_tokens=False, 
        special_tokens=SPECIAL_TOKENS,
        separations=None,
        separation_average=False,
        emphasis=None,
    ):
    # separations contains the index of starting tokens of each separation and the ending index of the last separation
    # if separation is specified, all attentions outside the separation will be ignored
    cutoff=attentions[min(token_ids_to_average)][0].shape[-1]
    chosen_attn = []
    for token in token_ids_to_average:
        if token == 0:
            attention = [layer[:,:,-1:,:] for layer in attentions[token]]
        else:
            attention = attentions[token]
        chosen_attn += [torch.stack([attention[i][:,:,:,:cutoff] for i in attn_layers], dim=0),]
    chosen_attn = torch.stack(chosen_attn, dim=0)
    average_attn = torch.mean(chosen_attn, dim=[0,1,2,3,4])
    if len(tokens) > average_attn.shape[0]:
        # print(f"Too much tokens: {len(tokens)} tokens. Cutting off the extra tokens to {cutoff} to fit the attention")
        tokens = tokens[:average_attn.shape[0]]
    if ignore_special_tokens and special_tokens is not None:
        for i in range(average_attn.shape[0]):
            average_attn[i]=0 if tokens[i] in special_tokens else average_attn[i]
    tokens = clean_word(tokens)    
    tokens = convert_special_tokens(tokens, special_tokens)
    preprocessed_attn = average_attn
    if separations is not None:
        attn_mask = [[separation[0], separation[-1]] for separation in separations]
        if any(r1[1] >= r2[0] and r1[0] <= r2[1] for i, r1 in enumerate(attn_mask) for r2 in attn_mask[i+1:]):
            raise Exception(f"There exist overlaps in the provided ranges, the ranges provided are {attn_mask}")
        result_attn = torch.zeros(preprocessed_attn.shape)
        for mask in attn_mask:
            result_attn[mask[0]:mask[1]] = preprocessed_attn[mask[0]:mask[1]]
        if emphasis is not None:
            result_attn = torch.exp(emphasis*result_attn/torch.sum(result_attn))
        for separation in separations:
            if len(separation) < 2:
                raise Exception(f"Separation should contain at least 2 elements: start and end. Separation provided is {separation}")
            if max(separation) > result_attn.shape[-1] or min(separation) < 0:
                raise Exception(f"Specified separation: {separation} is not contained in attention")
            start = separation[0]
            for end in separation[1:]:
                if separation_average:
                    result_attn[start:end] = torch.mean(result_attn[start:end])
                start = end
    normalized_attn = (result_attn / torch.sum(result_attn)).tolist()
    return normalized_attn

def vis_attn(
        attentions: Tuple[Tuple[torch.Tensor]],
        tokens: List[int],
        attn_layers: List[int], 
        words, 
        output_file, 
        ignore_special_tokens=False, 
        special_tokens=None,
        separations=None,
        separation_average=False,
        emphasis=None,
    ):
    normalized_attn, words = calc_attn(
        attentions, 
        tokens, 
        attn_layers, 
        words, 
        ignore_special_tokens, 
        special_tokens,
        separations,
        separation_average,
        emphasis,
    )
    generate(words, normalized_attn, output_file, color='purple', rescale_value=True)


# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2019-03-29 16:10:23
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-04-12 09:56:12


## convert the text/attention list to latex code, which will further generates the text heatmap based on attention weights.
import numpy as np

latex_special_token = ["!@#$%^&*()"]

def generate(text_list, attention_list, latex_file, color='red', rescale_value = False):
	assert(len(text_list) == len(attention_list))
	if rescale_value:
		attention_list = rescale(attention_list)
	word_num = len(text_list)
	text_list = clean_word(text_list)
	with open(latex_file,'w') as f:
		f.write(r'''\documentclass[varwidth]{standalone}
\special{papersize=210mm,297mm}
\usepackage{color}
\usepackage{tcolorbox}
\usepackage{CJK}
\usepackage{adjustbox}
\tcbset{width=0.9\textwidth,boxrule=0pt,colback=red,arc=0pt,auto outer arc,left=0pt,right=0pt,boxsep=5pt}
\begin{document}
\begin{CJK*}{UTF8}{gbsn}'''+'\n')
		string = r'''{\setlength{\fboxsep}{0pt}\colorbox{white!0}{\parbox{0.9\textwidth}{'''+"\n"
		for idx in range(word_num):
			formatted_number = "{:.10f}".format(attention_list[idx])
			string += "\\colorbox{%s!%s}{"%(color, formatted_number)+"\\strut " + text_list[idx]+"} "
		string += "\n}}}"
		f.write(string+'\n')
		f.write(r'''\end{CJK*}
\end{document}''')

def rescale(input_list):
	the_array = np.asarray(input_list)
	the_max = np.max(the_array)
	the_min = np.min(the_array)
	rescale = (the_array - the_min)/(the_max-the_min)*100
	return rescale.tolist()

def clean_word(word_list):
	new_word_list = []
	for word in word_list:
		for latex_sensitive in ["\\", "%", "&", "^", "#", "_",  "{", "}"]:
			if latex_sensitive in word:
				word = word.replace(latex_sensitive, '\\'+latex_sensitive)
		new_word_list.append(word)
	return new_word_list