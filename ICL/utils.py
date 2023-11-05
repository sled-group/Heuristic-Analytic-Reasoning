# This file has some generic utility functions
import datetime
import numpy as np
from matplotlib import pyplot as plt
import requests
from PIL import Image
from io import BytesIO
import random
import math
import os
import pickle

def load_glove_vectors(glove_file="glove.6B/glove.6B.50d.txt"):
    """
    Load GloVe vectors from a file.

    Args:
    glove_file (str): The path to the GloVe file.

    Returns:
    dict: A dictionary where the keys are the words and the values are the
          corresponding GloVe vectors.
    """
    cache_file = glove_file.replace('.txt', '.pkl')
    if os.path.isfile(cache_file):
        with open(cache_file, 'rb') as f:
            word_vectors = pickle.load(f)
    else:
        word_vectors = {}
        with open(glove_file, 'r', encoding='utf8') as f:
            for line in f:
                split_line = line.strip().split(' ')
                word = split_line[0]
                vector = [float(val) for val in split_line[1:]]
                word_vectors[word] = vector
        with open(cache_file, 'wb') as f:
            pickle.dump(word_vectors, f)
    return word_vectors

def glove_sequence_embedding(glove, emb_str):
  embs = emb_str.split()
  embs = [glove[w.lower()] if w.lower() in glove else np.zeros_like(glove['yes']) for w in embs]
  return np.mean(embs, axis=0)   

def glove_similarity_match(glove, query_str, search_strs, return_idx=True):
  query_emb = glove_sequence_embedding(glove, query_str)
  max_similarity = -100
  best_match = None
  best_match_idx = None
  for search_str_idx, search_str in enumerate(search_strs):
    search_emb = glove_sequence_embedding(glove, search_str)
    sim = np.dot(query_emb, search_emb)
    if sim > max_similarity:
        max_similarity = sim
        best_match = search_str
        best_match_idx = search_str_idx
  if return_idx:
    return best_match_idx
  else:
    return best_match

def get_one_hot(labels, c):
  if not isinstance(labels, list):
    labels = [labels]
  
  r = np.zeros(c)
  for l in labels:
    r[l] = 1
  return r

def softmax(l):
  l_sum = np.sum(np.exp(l))
  lr = []
  for li in l:
    lr.append(np.exp(li) / l_sum)
  return lr

def imload(url):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    response = requests.get(url)
    pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    # # convert to BGR format
    # image = np.array(pil_image)[:, :, [2, 1, 0]]
    return pil_image

def imopen(fname):
    pil_image = Image.open(fname)
    return pil_image

def imshow(img, caption):
    img = np.asarray(img)
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.figtext(0.5, 0.09, caption, wrap=True, horizontalalignment='center', fontsize=20)

def randomly_split(l, partition_sizes):
  assert sum(partition_sizes) == 1.0, "randomly_split partition sizes must add up to 1!"
  
  l = l.copy()
  random.shuffle(l)

  partitions = []
  boundary_old = 0
  for idx, ps in enumerate(partition_sizes):
    boundary = boundary_old + int(math.ceil(ps * len(l)))
    if idx != len(partition_sizes) - 1:
      partitions.append(l[boundary_old:boundary])
    else:
      partitions.append(l[boundary_old:])

    if ps > 0.0:
      assert len(partitions[-1]) > 0
    boundary_old = boundary

  assert sum([len(p) for p in partitions]) == len(l), "Randomly split list is missing something or has duplicates"
  return partitions

def get_output_dir_name(task, args, timestamp, run: int, result_type: str=None):
  # Unique instance name
  instance_name = ""
  if not result_type:
    if getattr(args, 'imagination_textual', None):
      if args.imagination_textual is not None:
          instance_name += "_" + args.imagination_textual
      if args.imagination_visual is not None:
          instance_name += "_" + args.imagination_visual
      if args.imagination_textual is not None or args.imagination_visual is not None:
          instance_name += "_k" + str(args.imagination_k)
          instance_name = "imagination" + instance_name
    else:
        instance_name = "no_imagination"
  else:
     instance_name = result_type
  if getattr(args, 'input_mode', None):
     instance_name += "_" + args.input_mode
  if getattr(args, 'attn_only', None) and args.attn_only:
    instance_name += "_attn_only"
  if getattr(args, 'reasoning_direction', None):
     instance_name += "_" + args.reasoning_direction
  if getattr(args, 'reasoning_depth', None):
     instance_name += "_" + args.reasoning_depth
  if getattr(args, 'use_conflict_explanations', None):
     if args.use_conflict_explanations:
      instance_name += "_" + "expl"  
  if args.debug:
    instance_name += "_debug"
  instance_name += "_" + timestamp.strftime("%Y%m%d%H%M%S")

  # Model name folder
  if getattr(args, 'mm_backbone', None):
    model_name = args.mm_backbone
  elif getattr(args, 'lm_backbone', None):
    model_name = args.lm_backbone
  else:
    raise ValueError("Can't find model name to generate output dir name.")
  
  if getattr(args, 'demo_choice', None):
    demo_choice = args.demo_choice
  else:
    raise ValueError("Can't find demo choice name to generate output dir name.")
  
  # Build full output directory
  output_dir = os.path.join(args.output_dir, task, model_name.replace('/','_'), demo_choice.replace('/','_'),instance_name)
  if getattr(args, 'n_runs', None) and args.n_runs > 1:
    output_dir = os.path.join(output_dir, f'run{run}')
  return output_dir, instance_name

def get_black_image(width, height):
  return Image.fromarray(np.zeros([width,height,3],dtype=np.uint8))

def crop_image_to_square(image):
   if image.width > image.height:
      center_w = image.width // 2
      left_bound = center_w - image.height / 2 + (0.5 if image.height % 2 != 0 else 0.0) # Shift crop by 0.5 pixels if odd number
      right_bound = center_w + image.height / 2 + 1 + (0.5 if image.height % 2 != 0 else 0.0)
      return image.crop((left_bound, 0, right_bound, image.height-1))
   else:
      raise NotImplementedError("This function can only handle landscape shaped images right now.")
      

def cosine_similarity(v1, v2):
   return (v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))