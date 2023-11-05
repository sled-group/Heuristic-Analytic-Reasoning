"""Some code borrowed from PIGLeT code base https://github.com/rowanz/piglet/blob/main/data/thor_constants.py"""
import pickle
import os
import pandas as pd
from PIL import Image
from typing import List
import numpy as np
from tqdm import tqdm
from string import Formatter
import sys

PIGLET_PATH = '/nfs/turbo/coe-chaijy/physical_commonsense/piglet/pretraining_data/'
THOR_OBJECT_TYPE_TO_IND = {
 'background': 0,
 'alarm clock': 1,
 'aluminum foil': 2,
 'apple': 3,
 'apple sliced': 4,
 'arm chair': 5,
 'baseball bat': 6,
 'basketball': 7,
 'bathtub': 8,
 'bathtub basin': 9,
 'bed': 10,
 'bench': 11,
 'blinds': 12,
 'book': 13,
 'boots': 14,
 'bottle': 15,
 'bowl': 16,
 'box': 17,
 'bread': 18,
 'sliced bread': 19,
 'butter knife': 20,
 'CD': 21,
 'cabinet': 22,
 'candle': 23,
 'cell phone': 24,
 'chair': 25,
 'cloth': 26,
 'coffee machine': 27,
 'coffee table': 28,
 'countertop': 29,
 'credit card': 30,
 'cup': 31,
 'curtains': 32,
 'desk': 33,
 'desk lamp': 34,
 'desktop': 35,
 'dining table': 36,
 'dish sponge': 37,
 'dog bed': 38,
 'drawer': 39,
 'dresser': 40,
 'dumbbell': 41,
 'egg': 42,
 'cracked egg': 43,
 'faucet': 44,
 'floor': 45,
 'floor lamp': 46,
 'footstool': 47,
 'fork': 48,
 'fridge': 49,
 'garbage bag': 50,
 'garbage can': 51,
 'hand towel': 52,
 'hand towel holder': 53,
 'house plant': 54,
 'kettle': 55,
 'keychain': 56,
 'knife': 57,
 'ladle': 58,
 'laptop': 59,
 'laundry hamper': 60,
 'lettuce': 61,
 'sliced lettuce': 62,
 'light switch': 63,
 'microwave': 64,
 'mirror': 65,
 'mug': 66,
 'newspaper': 67,
 'ottoman': 68,
 'painting': 69,
 'pan': 70,
 'paper towel roll': 71,
 'pen': 72,
 'pencil': 73,
 'pepper shaker': 74,
 'pillow': 75,
 'plate': 76,
 'plunger': 77,
 'poster': 78,
 'pot': 79,
 'potato': 80,
 'sliced potato': 81,
 'remote control': 82,
 'room decor': 83,
 'safe': 84,
 'salt shaker': 85,
 'scrub brush': 86,
 'shelf': 87,
 'shelving unit': 88,
 'shower curtain': 89,
 'shower door': 90,
 'shower glass': 91,
 'shower head': 92,
 'side table': 93,
 'sink': 94,
 'sink basin': 95,
 'soap bar': 96,
 'soap bottle': 97,
 'sofa': 98,
 'spatula': 99,
 'spoon': 100,
 'spray bottle': 101,
 'statue': 102,
 'stool': 103,
 'stove burner': 104,
 'stove knob': 105,
 'TV stand': 106,
 'table top decor': 107,
 'teddy bear': 108,
 'television': 109,
 'tennis racket': 110,
 'tissue box': 111,
 'toaster': 112,
 'toilet': 113,
 'toilet paper': 114,
 'toilet paper hanger': 115,
 'tomato': 116,
 'sliced tomato': 117,
 'towel': 118,
 'towel holder': 119,
 'vacuum cleaner': 120,
 'vase': 121,
 'watch': 122,
 'watering can': 123,
 'window': 124,
 'wine bottle': 125,
}
THOR_OBJECT_IND_TO_TYPE = {v: k  for k, v in THOR_OBJECT_TYPE_TO_IND.items()}
MASS_CATEGORIES = [
    (-1e8, 1e-8),  # Massless -- 46.5 %
    (1e-8, 0.04500),
    (0.04500, 0.11000),
    (0.11000, 0.24000),
    (0.24000, 0.62000),
    (0.62000, 1.00000),
    (1.00000, 5.00000),
    (5.00000, float('inf')),
]
SIZE_CATEGORIES = [
 (-1e-08, 3.7942343624308705e-05),
 (3.7942343624308705e-05, 0.0006833796505816281),
 (0.0006833796505816281, 0.0028182819951325655),
 (0.0028182819951325655, 0.010956133715808392),
 (0.010956133715808392, 0.038569377816362015),
 (0.038569377816362015, 0.08704059571027756),
 (0.08704059571027756, 0.18317937850952148),
 (0.18317937850952148, float('inf')),
]
DISTANCE_CATEGORIES = [(-1.0, 0.25),
 (0.25, 0.5),
 (0.5, 0.75),
 (0.75, 1.0),
 (1.0, 1.5),
 (1.5, 2.0),
 (2.0, 2.5),
 (2.5, float('inf')),
]
# [affordance_name, arity, is_object]
PIGLET_ATTRIBUTES = {
  'ObjectTemperature': {'arity': 3, 'is_object': False},
  'breakable': {'arity': 1, 'is_object': False},
  'cookable': {'arity': 1, 'is_object': False},
  'dirtyable': {'arity': 1, 'is_object': False},
  'isBroken': {'arity': 1, 'is_object': False},
  'isCooked': {'arity': 1, 'is_object': False},
  'isDirty': {'arity': 1, 'is_object': False},
  'isFilledWithLiquid': {'arity': 1, 'is_object': False},
  'isOpen': {'arity': 1, 'is_object': False},
  'isPickedUp': {'arity': 1, 'is_object': False},
  'isSliced': {'arity': 1, 'is_object': False},
  'isToggled': {'arity': 1, 'is_object': False},
  'isUsedUp': {'arity': 1, 'is_object': False},
  'mass': {'arity': len(MASS_CATEGORIES), 'is_object': False},
  'size': {'arity': len(SIZE_CATEGORIES), 'is_object': False},
  'distance': {'arity': len(DISTANCE_CATEGORIES), 'is_object': False},
  'moveable': {'arity': 1, 'is_object': False},
  'openable': {'arity': 1, 'is_object': False},
  'parentReceptacles': {'arity': len(THOR_OBJECT_TYPE_TO_IND), 'is_object': True},
  'pickupable': {'arity': 1, 'is_object': False},
  'receptacle': {'arity': 1, 'is_object': False},
  'receptacleObjectIds': {'arity': len(THOR_OBJECT_TYPE_TO_IND), 'is_object': True},
  'salientMaterials_Ceramic': {'arity': 1, 'is_object': False},
  'salientMaterials_Fabric': {'arity': 1, 'is_object': False},
  'salientMaterials_Food': {'arity': 1, 'is_object': False},
  'salientMaterials_Glass': {'arity': 1, 'is_object': False},
  'salientMaterials_Leather': {'arity': 1, 'is_object': False},
  'salientMaterials_Metal': {'arity': 1, 'is_object': False},
  'salientMaterials_Organic': {'arity': 1, 'is_object': False},
  'salientMaterials_Paper': {'arity': 1, 'is_object': False},
  'salientMaterials_Plastic': {'arity': 1, 'is_object': False},
  'salientMaterials_Rubber': {'arity': 1, 'is_object': False},
  'salientMaterials_Soap': {'arity': 1, 'is_object': False},
  'salientMaterials_Sponge': {'arity': 1, 'is_object': False},
  'salientMaterials_Stone': {'arity': 1, 'is_object': False},
  'salientMaterials_Wax': {'arity': 1, 'is_object': False},
  'salientMaterials_Wood': {'arity': 1, 'is_object': False},
  'sliceable': {'arity': 1, 'is_object': False},
  'toggleable': {'arity': 1, 'is_object': False},
}

THOR_ACTIONS = ['CloseObject',
                'DirtyObject',
                'EmptyLiquidFromObject',
                'Done',
                'JumpAhead',
                'LookDown',
                'LookUp',
                'MoveAhead',
                'MoveBack',
                'MoveLeft',
                'MoveRight',
                'OpenObject',
                'PickupObject',
                'PutObject',
                'RotateHand',
                'RotateLeft',
                'RotateRight',
                'SliceObject',
                'ThrowObject10',  # Several different magnitudes
                'ThrowObject100',
                'ThrowObject1000',
                'ToggleObjectOff',
                'ToggleObjectOn',
                'HeatUpPan', # hack
                ]
THOR_ACTION_TYPE_TO_IND = {t: i for i, t in enumerate(['__background__'] + THOR_ACTIONS)}
THOR_ACTION_IND_TO_TYPE = {v: k for k, v in THOR_ACTION_TYPE_TO_IND.items()}
THOR_ACTIONS = ['CloseObject',
                'DirtyObject',
                'EmptyLiquidFromObject',
                'Done',
                'JumpAhead',
                'LookDown',
                'LookUp',
                'MoveAhead',
                'MoveBack',
                'MoveLeft',
                'MoveRight',
                'OpenObject',
                'PickupObject',
                'PutObject',
                'RotateHand',
                'RotateLeft',
                'RotateRight',
                'SliceObject',
                'ThrowObject10',  # Several different magnitudes
                'ThrowObject100',
                'ThrowObject1000',
                'ToggleObjectOff',
                'ToggleObjectOn',
                'HeatUpPan', # hack
                ]

# Names for each value of each class 
# (this is a subset of above, as we'll only evaluate on attributes that can change, not affordances/materials of object)
PIGLET_ATTRIBUTES_NAMES = {
  'ObjectTemperature': ['cold', 'room temperature', 'hot'],
  'isBroken': ['functional', 'broken'],
  'isCooked': ['uncooked', 'cooked'],
  'isDirty': ['clean', 'dirty'],
  'isFilledWithLiquid': ['empty', 'filled'],
  'isOpen': ['closed', 'open'],
  'isPickedUp': ['there', 'in hand'], # TODO: find a better prompt word than 'there'
  'isSliced': ['whole', 'sliced'],
  'isToggled': ['turned off', 'turned on'],
  'isUsedUp': ['not used up', 'used up'],
  'mass': ['massless', 'below 0.1 lb', '.1 to .2lb', '.2 to .5lb', '.5 to 1lb', '1 to 2lb', '2 to 10lb', 'over 10lb'],
  'size': ['sizeless', 'tiny', 'small', 'medium', 'medium-plus', 'large', 'extra large', 'extra extra large'],
  'distance': ['below 1ft away', '1 to 2ft away', '2 to 3ft away', '3 to 4ft away', '4 to 6ft away', '6 to 8 ft away', '8 to 10ft away', 'over 10ft away'],
  'parentReceptacles': list(THOR_OBJECT_TYPE_TO_IND.keys())
}


# Synthetic templates for pre-training data - only includes state change actions
PIGLET_ACTION_DESCRIPTION_TEMPLATES = {
  'CloseObject': [
    "The robot closes the {obj0}.",
    "The robot closes up the {obj0}.",
    "The robot pushes the {obj0} closed."
  ],
  'DirtyObject': [
    "The robot dirties the {obj0}.",
    "The robot soils the {obj0}.",
    "The robot makes the {obj0} dirty."
  ],
  'EmptyLiquidFromObject': [
    "The robot empties the {obj0}.",
    "The robot pours out the {obj0}.",
    "The robot dumps out the {obj0}.",
  ],
  'OpenObject': [
    "The robot opens the {obj0}.",
    "The robot opens up the {obj0}.",
    "The robot pulls open the {obj0}."
  ],
  'PickupObject': [
    "The robot picks up the {obj0}.",
    "The {obj0} gets picked up.",
    "The robot grabs the {obj0}."
  ],
  'PutObject': [
    "The robot places the {obj0} {prep0} the {obj1}.",
    "The robot puts the {obj0} {prep0} the {obj1}.",
    "The robot sets the {obj0} {prep0} the {obj1}.",
  ],
  'SliceObject': [
    "The robot slices the {obj0}.",
    "The robot cuts up the {obj0}.",
    "The robot chops up the {obj0}.",
  ],
  'ThrowObject10': [
    "The robot tosses the {obj0}.",
    "The robot throws the {obj0}.",
    "The robot drops the {obj0}.",
  ],
  'ThrowObject100': [
    "The robot tosses the {obj0}.",
    "The robot throws the {obj0}.",
    "The robot launches the {obj0}.",
  ],
  'ThrowObject1000': [
    "The robot tosses the {obj0}.",
    "The robot throws the {obj0}.",
    "The robot flings the {obj0}.",
  ],
  'ToggleObjectOff': [
    "The robot turns off the {obj0}.",
    "The robot toggles off the {obj0}.",
    "The robot turns the {obj0} off.",
  ],
  'ToggleObjectOn': [
    "The robot turns on the {obj0}.",
    "The robot toggles on the {obj0}.",
    "The robot turns the {obj0} on.",
  ],
  'HeatUpPan': [
    "The robot heats up the {obj0}.",
    "The robot heats the {obj0}.",
    "The robot starts heating the {obj0}."
  ],
  "CoolObject": [
    "The robot cools the {obj0}.",
    "The robot cools down the {obj0}.",
    "The robot chills the {obj0}."
  ],
  "BreakObject": [
    "The robot breaks the {obj0}.",
    "The robot smashes the {obj0}.",
    "The robot destroys the {obj0}.",
  ],
  "BreakObject": [
    "The robot breaks the {obj0}.",
    "The robot smashes the {obj0}.",
    "The robot destroys the {obj0}.",
  ],
  "RepairObject": [
    "The robot repairs the {obj0}.",
    "The robot fixes the {obj0}.",
    "The robot fixes up the {obj0}.",
  ],
  "CookObject": [
    "The robot cooks the {obj0}.",
    "The robot heats up the {obj0}.",
    "The robot cooks up the {obj0}.",
  ],
  "CleanObject": [
    "The robot cleans the {obj0}.",
    "The robot rinses the {obj0}.",
    "The robot washes the {obj0}.",
  ],
  "FillObject": [
    "The robot fills up the {obj0}.",
    "The robot fills the {obj0}.",
    "The robot fills the {obj0} up."
  ],
  "UseUpObject": [
    "The robot uses up the {obj0}.",
    "The robot uses the rest of the {obj0}.",
    "The robot finishes the {obj0}.",
  ],
}

THOR_ACTION_TO_OBJ_STATES = {
  "CloseObject": [
    ("isOpen", True, False)
  ],
  "DirtyObject": [
    ("isDirty", True, False)
  ],
  "EmptyLiquidFromObject": [
    ("isFilledWithLiquid", True, False)
  ],
  'OpenObject': [
    ("isOpen", False, True)
  ],
  'PickupObject': [
    ("isPickedUp", False, True)
  ],
  'PutObject': [
    ("isPickedUp", True, False)
  ],
  'SliceObject': [
    ("isSliced", False, True)
  ],
  'ThrowObject10': [
    ("isPickedUp", True, False)
  ],
  'ThrowObject100': [
    ("isPickedUp", True, False)
  ],
  'ThrowObject1000': [
    ("isPickedUp", True, False)
  ],
  'ToggleObjectOff': [
    ("isToggled", True, False)
  ],
  'ToggleObjectOn': [
    ("isToggled", False, True),
  ],
  'HeatUpPan': [
    ("ObjectTemperature", 0, 2),
    ("ObjectTemperature", 1, 2),
  ]
}

# Some other possible action-like phrases a human could say
# based on objects' state changes
STATE_BASED_ACTIONS = {
   ("ObjectTemperature", 2, 1): "CoolObject",
   ("ObjectTemperature", 2, 0): "CoolObject",
   ("ObjectTemperature", 1, 0): "CoolObject",
   ("isBroken", False, True): "BreakObject",
   ("isBroken", True, False): "RepairObject",
   ("isCooked", False, True): "CookObject",
   ("isDirty", True, False): "CleanObject",
   ("isFilledWithLiquid", False, True): "FillObject",
   ("isUsedUp", False, True): "UseUpObject",
}

IN_OBJECTS = {
 'bathtub',
 'bathtub basin',
 'bowl',
 'box',
 'cabinet',
 'coffee machine',
 'cup',
 'drawer',
 'fridge',
 'garbage can',
 'house plant',
 'kettle',
 'laundry hamper',
 'microwave',
 'mug',
 'pan',
 'pot',
 'safe',
 'sink',
 'sink basin',
 'toaster',
 'toilet',
 'vase',
 'watering can',
}


MMPP_DIR = '/nfs/turbo/coe-chaijy/physical_commonsense/piglet'

# Crop piglet image into precondition and postcondition images
# Returns precondition and postcondition images
def crop_piglet_image(img_loc):
  im = Image.open(img_loc)

  width, height = im.size
  
  left1 = 0
  top1 = 0
  right1 = width
  bottom1 = height / 2

  left2 = 0
  top2 = height / 2
  right2 = width
  bottom2 = height
  
  # Cropped image of above dimension
  # (It will not change original image)
  im_pre = im.crop((left1, top1, right1, bottom1))
  im_post = im.crop((left2, top2, right2, bottom2))  

  return im_pre, im_post

def label_physical_states(states: List[int]):
  states = {
    attr: state for attr, state in zip(PIGLET_ATTRIBUTES, states)
  }
  states = {
    k: v for k, v in states.items() if k in PIGLET_ATTRIBUTES_NAMES
  }
  return states

def load_piglet_dataset():
  PIGLET_CACHE_FILE = 'cache_files/piglet.pkl'
  if os.path.exists(PIGLET_CACHE_FILE):
    piglet_lang_image = pickle.load(open(PIGLET_CACHE_FILE, 'rb'))
    return piglet_lang_image

  piglet_lang_image = {'train': [], 'val': [], 'test': []}

  piglet_df = pd.read_csv(PIGLET_PATH + "lang_image_matched.csv", lineterminator='\n')

  # iterate over each group
  split_df = piglet_df.groupby(['split'])
  for split, df_group in split_df:
      for _, row in df_group.iterrows():
        if '[' in row['object_names']:
          object_names = eval(row['object_names'])
        else:
          object_names = [row['object_names']]
        im_pre, im_post = crop_piglet_image(row['image_locations'])
        for oi, object_name in enumerate(object_names):
          precondition_states = np.array(eval(row['pre_state'])).reshape(len(PIGLET_ATTRIBUTES), len(object_names))
          effect_states = np.array(eval(row['post_state'])).reshape(len(PIGLET_ATTRIBUTES), len(object_names))
          if len(precondition_states.shape) == 1:
            precondition_states = np.expand_dims(precondition_states, axis=1)
            effect_states = np.expand_dims(effect_states, axis=1)

          precondition_states = precondition_states[:, oi]
          effect_states = effect_states[:, oi]
          assert len(precondition_states) == len(effect_states) == len(PIGLET_ATTRIBUTES)

          precondition_states = label_physical_states(precondition_states)
          effect_states = label_physical_states(effect_states)
          action_name = THOR_ACTION_IND_TO_TYPE[row['action_id']]

          piglet_lang_image[split].append(
            { # loading all three annotations
              'image_id': row['image_id'],
              'entity': object_name,
              'action_name': action_name,
              'precondition_language': row['precondition_language'],
              'action_language': row['action_language'],
              'effect_language': row['postcondition_language'],
              'precondition_image': im_pre.convert('RGB'),
              'effect_image': im_post.convert('RGB'),
              'precondition_states': precondition_states,
              'effect_states': effect_states,
            })
            
  # Change dict keys to train, dev, test
  piglet_lang_image['dev'] = piglet_lang_image.pop('val')

  pickle.dump(piglet_lang_image, open(PIGLET_CACHE_FILE, 'wb'))
  return piglet_lang_image

def load_piglet_pretraining_dataset(target_split):
  piglet_downstream_dataset = load_piglet_dataset()
  downstream_dev_test_ids = [ex['image_id'] for ex in piglet_downstream_dataset['dev']] + \
                            [ex['image_id'] for ex in piglet_downstream_dataset['test']]

  piglet_df = pd.read_csv(PIGLET_PATH + "image.csv", lineterminator='\n')

  # iterate over each group
  split_df = piglet_df.groupby(['split'])
  for split, df_group in split_df:
      if split != target_split:
         continue
      for _, row in df_group.iterrows():
        # Don't use any dev or test images for training diffusion models
        if row['image_id'] in downstream_dev_test_ids:
           continue

        object_names = [THOR_OBJECT_IND_TO_TYPE[idx] for idx in eval(row['object_types'])]
        im_pre, im_post = crop_piglet_image(row['image_locations'])
        for oi, object_name in enumerate(object_names):
          precondition_states = np.array(eval(row['pre_state'])).reshape(len(PIGLET_ATTRIBUTES), len(object_names))
          effect_states = np.array(eval(row['post_state'])).reshape(len(PIGLET_ATTRIBUTES), len(object_names))
          if len(precondition_states.shape) == 1:
            precondition_states = np.expand_dims(precondition_states, axis=1)
            effect_states = np.expand_dims(effect_states, axis=1)

          precondition_states = precondition_states[:, oi]
          effect_states = effect_states[:, oi]
          assert len(precondition_states) == len(effect_states) == len(PIGLET_ATTRIBUTES)

          # Restrict to evaluation attributes
          precondition_states = label_physical_states(precondition_states)
          effect_states = label_physical_states(effect_states)

          # Generate candidate action language based on states and action
          candidate_action_language = []

          # First look for something to say based on the action
          action_name = THOR_ACTION_IND_TO_TYPE[row['action_id']]
          for action_attribute, action_pre, action_eff in THOR_ACTION_TO_OBJ_STATES[action_name]:
            if precondition_states[action_attribute] == action_pre and \
              effect_states[action_attribute] == action_eff:
                for template in PIGLET_ACTION_DESCRIPTION_TEMPLATES[action_name]:
                  arg_names = [fn for _, fn, _, _ in Formatter().parse(template) if fn is not None]
                  kwargs = {}
                  if 'obj0' in arg_names:
                    kwargs['obj0'] = object_name
                  if 'obj1' in arg_names:
                    # Other object should be the indirect object
                    kwargs['obj1'] = object_names[1 - oi]
                  if 'prep0' in arg_names:
                    assert 'obj1' in kwargs
                    kwargs['prep0'] = 'in' if object_name in IN_OBJECTS else 'on'
                  candidate_action_language.append(template.format(**kwargs))
              
          # Then look at the state changes to see if we can say anything else
          for attribute, pre, eff in STATE_BASED_ACTIONS:
            if precondition_states[attribute] == pre and \
               effect_states[attribute] == eff:
              state_action_name = STATE_BASED_ACTIONS[(attribute, pre, eff)]
              for template in PIGLET_ACTION_DESCRIPTION_TEMPLATES[state_action_name]:
                candidate_action_language.append(template.format(obj0=object_name))

          # Describe precondition states based on states that changed
          candidate_precondition_language = [
            "There is {article0} {obj0}.".format(article0='an' if object_name[0].lower() in 'aeiou' else 'a', 
                                                 obj0=object_name)
          ]
          candidate_effect_language = []
          for attribute in PIGLET_ATTRIBUTES_NAMES:
            if precondition_states[attribute] != effect_states[attribute]:
              if attribute != 'parentReceptacles':
                pre_template = "The {obj0} is {state0}."
                candidate_precondition_language.append(pre_template.format(obj0=object_name,
                                                                           state0=PIGLET_ATTRIBUTES_NAMES[attribute][int(precondition_states[attribute])]))
                eff_template = "The {obj0} is now {state0}."
                candidate_effect_language.append(eff_template.format(obj0=object_name,
                                                                state0=PIGLET_ATTRIBUTES_NAMES[attribute][int(effect_states[attribute])]))
              else:
                pre_template = "The {obj0} is {prep0} the {obj1}."
                if precondition_states[attribute] > 0:
                  obj1 = PIGLET_ATTRIBUTES_NAMES[attribute][int(precondition_states[attribute])]
                  candidate_precondition_language.append(pre_template.format(obj0=object_name,
                                                                            prep0='in' if obj1 in IN_OBJECTS else 'on',
                                                                            obj1=obj1))
                eff_template = "The {obj0} is now {prep0} the {obj1}."
                if effect_states[attribute] > 0:
                  obj1 = PIGLET_ATTRIBUTES_NAMES[attribute][int(effect_states[attribute])]
                  candidate_effect_language.append(pre_template.format(obj0=object_name,
                                                                            prep0='in' if obj1 in IN_OBJECTS else 'on',
                                                                            obj1=obj1))

          # Only return examples for entities where we produced action language                 
          if len(candidate_action_language) > 0:
            record = { # loading all three annotations
                'image_id': row['image_id'],
                'entity': object_name,
                'action_name': action_name,
                'precondition_language': candidate_precondition_language,
                'action_language': candidate_action_language,
                'effect_language': candidate_effect_language,
                'precondition_image': im_pre.convert('RGB'),
                'effect_image': im_post.convert('RGB'),
                'precondition_states': precondition_states,
                'effect_states': effect_states,
              }
            yield record
