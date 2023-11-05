# This file has task data preprocessing functions that may be useful for any task
from datasets import load_dataset
import numpy as np
import pickle
import os
import spacy
import requests
from tqdm import tqdm
import random
import pandas as pd
from PIL import Image
import json
import mlconjug3

from utils import randomly_split

# TODO: fix compatibility of MMPP environment with spacy code

nlp = spacy.load("en_core_web_sm", disable=["parser", "attribute_ruler", "lemmatizer", "ner"])

def add_indefinite_article(noun):
  if is_plural(noun):
    return noun
  else:
    if noun[0].lower() in ["a", "e", "i", "o", "u"]:
      return "an " + noun
    else:
      return "a " + noun

def is_plural(noun):
  doc = nlp("I have found the %s" % noun) # Some artificial prompt for sense of word to be taken as noun
  for token in doc:
    if token.text == noun:
      if token.tag_ in ["NNS", "NNPS"]:
        return True
      else:
        return False
  
  # If first part failed, use heuristic
  if (noun.endswith("s") and not noun.endswith("ss")) or noun.endswith("es"):
    return True
  else:
    return False

# The below tense conversion functions are a bit slow, so we cache any responses generated with them

# Convert present-tense verb to past tense
CACHE_VERB_PAST_TENSE_FILE = os.path.join("cache_files", "verb_past_tense_cache.json")
CACHE_VERB_PAST_TENSE = json.load(open(CACHE_VERB_PAST_TENSE_FILE, "r")) if os.path.exists(CACHE_VERB_PAST_TENSE_FILE) else {}
def verb_past_tense(verb):
  if verb in CACHE_VERB_PAST_TENSE:
    return CACHE_VERB_PAST_TENSE[verb]
  else:
    verb_conjugator = mlconjug3.Conjugator(language='en')
    conjugated_verb = verb_conjugator.conjugate(verb).conjug_info['indicative']['indicative past tense']['3s'].split('/')[0]
    CACHE_VERB_PAST_TENSE[verb] = conjugated_verb
    json.dump(CACHE_VERB_PAST_TENSE, open(CACHE_VERB_PAST_TENSE_FILE, "w"))
    return conjugated_verb


CACHE_VERB_THIRD_PERSON_SINGULAR_PRESENT_TENSE_FILE = os.path.join("cache_files", "verb_third_person_singular_present_tense_cache.json")
CACHE_VERB_THIRD_PERSON_SINGULAR_PRESENT_TENSE = json.load(open(CACHE_VERB_THIRD_PERSON_SINGULAR_PRESENT_TENSE_FILE, "r")) if os.path.exists(CACHE_VERB_THIRD_PERSON_SINGULAR_PRESENT_TENSE_FILE) else {}
def verb_third_person_singular_present_tense(verb):
  """Conjugates verb to present tense, third person singular (e.g., "beat" -> "beats")."""
  if verb in CACHE_VERB_THIRD_PERSON_SINGULAR_PRESENT_TENSE:
    return CACHE_VERB_THIRD_PERSON_SINGULAR_PRESENT_TENSE[verb]
  else:
    verb_conjugator = mlconjug3.Conjugator(language='en')
    conjugated_verb = verb_conjugator.conjugate(verb).conjug_info['indicative']['indicative present']['3s'].split('/')[0]
    CACHE_VERB_THIRD_PERSON_SINGULAR_PRESENT_TENSE[verb] = conjugated_verb
    json.dump(CACHE_VERB_THIRD_PERSON_SINGULAR_PRESENT_TENSE, open(CACHE_VERB_THIRD_PERSON_SINGULAR_PRESENT_TENSE_FILE, "w"))
    return conjugated_verb