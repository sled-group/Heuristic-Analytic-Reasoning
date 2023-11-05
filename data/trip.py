import torch
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import pandas as pd
import json
from PIL import Image

from utils import randomly_split

def get_eff_attr_state_text(stories_info):
    implausible_story = None
    if stories_info[0]['plausible'] == True:
        implausible_story = stories_info[1]
    else:
        implausible_story = stories_info[0]
    confl_sentence_1_idx = implausible_story['confl_pairs'][0][0]
    confl_sentence_2_idx = implausible_story['confl_pairs'][0][1]
    candidates = []
    for entity_idx, entity in enumerate(implausible_story['entities']):
        effect_states = implausible_story['states'][entity_idx][confl_sentence_1_idx][1]
        precondition_states = implausible_story['states'][entity_idx][confl_sentence_2_idx][0]
        for state_i in range(0, 20):
            attribute = attr_all[state_i]
            eff_state = effect_states[state_i]
            pre_state = precondition_states[state_i]
            if (eff_state == 1 and pre_state == 2) or (eff_state == 2 and pre_state == 1):
                candidates.append({"entity": entity, "attribute": attribute, "eff_state": eff_state, "pre_state": pre_state})
    eff_attr_state_text = None
    for candidate in candidates:
        entity = candidate["entity"]
        attribute = candidate["attribute"]
        eff_state = candidate["eff_state"]
        pre_state = candidate["pre_state"]
        if attribute == "h_location":
            eff_attr_state_text = h_location_map[eff_state]
        elif attribute == "location":
            eff_attr_state_text = o_location_map[eff_state]
        else:
            eff_attr_state_text = other_attr_map[attribute][eff_state - 1]
    return eff_attr_state_text

def get_pre_attr_state_text(stories_info):
    implausible_story = None
    if stories_info[0]['plausible'] == True:
        implausible_story = stories_info[1]
    else:
        implausible_story = stories_info[0]
    confl_sentence_1_idx = implausible_story['confl_pairs'][0][0]
    confl_sentence_2_idx = implausible_story['confl_pairs'][0][1]
    candidates = []
    for entity_idx, entity in enumerate(implausible_story['entities']):
        effect_states = implausible_story['states'][entity_idx][confl_sentence_1_idx][1]
        precondition_states = implausible_story['states'][entity_idx][confl_sentence_2_idx][0]
        for state_i in range(0, 20):
            attribute = attr_all[state_i]
            eff_state = effect_states[state_i]
            pre_state = precondition_states[state_i]
            if (eff_state == 1 and pre_state == 2) or (eff_state == 2 and pre_state == 1):
                candidates.append({"entity": entity, "attribute": attribute, "eff_state": eff_state, "pre_state": pre_state})
    pre_attr_state_text = None
    for candidate in candidates:
        entity = candidate["entity"]
        attribute = candidate["attribute"]
        eff_state = candidate["eff_state"]
        pre_state = candidate["pre_state"]
        if attribute == "h_location":
            pre_attr_state_text = h_location_map[pre_state]
        elif attribute == "location":
            pre_attr_state_text = o_location_map[pre_state]
        else:
            pre_attr_state_text = other_attr_map[attribute][pre_state - 1]
    return pre_attr_state_text

def check_no_multi_confl(stories):
    implausible_story = None
    if stories[0]["plausible"] == False and stories[1]["plausible"] == True:
        implausible_story = stories[0]
    elif stories[1]["plausible"] == False and stories[0]["plausible"] == True:
        implausible_story = stories[1]
    else:
        return False
    if len(implausible_story["confl_pairs"]) == 1:
        return True
    else:
        return False

# TODO: Add changed_state with entity_name
def raw_dict_to_dataset(raw_dict, exclude_multi_confl=True, condense_multiple_conflict=False, reduce_options=False):
    attributes = ["h_location", "conscious", "wearing", "h_wet", "hygiene", "location", "exist", "clean", "power", "functional", "pieces", "wet", "open", "temperature", "solid", "contain", "running", "moveable", "mixed", "edible"]
    other_attributes_map = {0: [0, 0], 1:[1, 1], 2:[2, 2], 3:[2, 1], 4:[1, 2], 5:[0, 1], 6:[0, 2], 7:[1, 0], 8:[2, 0]}
    dataset = []
    multi_conflicts = 0
    for object in raw_dict:
        assert len(object["stories"]) == 2
        if len(object["stories"]) == 2 and "states" in object["stories"][0].keys() and "states" in object["stories"][1].keys(): # some stories in the test dataset doesn't have states
            if not(exclude_multi_confl) or check_no_multi_confl(object["stories"]):
                # p => q is logically equivalent to not(p) or q
                stories_json_data = {}
                stories = []
                have_all_states = True
                for story in object["stories"]:
                    if len(story['sentences']) != len(story['states']):
                        print('missed a story in example %s!' % object['example_id'])
                        have_all_states = False                  

                    story_json_data = {}
                    story_json_data["sentences"] = story["sentences"]
                    story_json_data["confl_pairs"] = story["confl_pairs"]
                    story_json_data["confl_sents"] = story["confl_sents"]
                    story_json_data["breakpoint"] = story["breakpoint"]
                    if len(story["confl_pairs"]) > 1:
                        if condense_multiple_conflict:
                            # Use heuristic to condense multiple conflict pairs into 1
                            story_json_data["confl_pairs"] = [condense_multiple_conflict_pairs(story_json_data["confl_sents"], story_json_data["breakpoint"])]
                        multi_conflicts += 1
                    if len(story_json_data["confl_pairs"]) == 0:
                        story_json_data["plausible"] = 1
                    else:
                        story_json_data["plausible"] = 0
                    entities = []
                    story_json_data['entities_by_sentence'] = []
                    for sentence_i in range(0, len(story["states"])):
                        entities_by_sentence = []
                        sentence_states = story["states"][sentence_i]
                        for attribute in sentence_states:
                            for entity_state in sentence_states[attribute]:
                                entity, state = entity_state
                                if entity not in entities:
                                    entities.append(entity)
                                if entity not in entities_by_sentence:
                                    entities_by_sentence.append(entity)
                        story_json_data['entities_by_sentence'].append(entities_by_sentence)
                    story_json_data["entities"] = entities
                    states = []
                    for _ in range(len(entities)):
                        sentences_state = []
                        for _ in range(len(story_json_data["sentences"])):
                            sentences_state.append([[0] * 20, [0] * 20])
                        states.append(sentences_state)
                    total_info_states = []
                    total_changed_states = []
                    total_confl_states = []
                    latest_states = {}
                    for sentence_i in range(0, len(story["sentences"])):
                        info_states = {}
                        confl_states = {}
                        changed_states = {}
                        if sentence_i < len(story["states"]): # handle incomplete data
                            sentence_states = story["states"][sentence_i]
                            for attribute in sentence_states:
                                for entity_state in sentence_states[attribute]:
                                    entity, state = entity_state
                                    if entity not in info_states.keys():
                                        info_states[entity] = []
                                    if attribute == "h_location" or attribute == "location":
                                        info_states[entity].append([attribute, state])
                                        states[entities.index(entity)][sentence_i][0][attributes.index(attribute)] = state
                                        states[entities.index(entity)][sentence_i][1][attributes.index(attribute)] = state
                                    else:
                                        if state != 0: # and state != 1 and state != 2:
                                            info_states[entity].append([attribute, other_attributes_map[state]])
                                        if state != 0 and state != 1 and state != 2:
                                            if entity not in changed_states.keys():
                                                changed_states[entity] = []
                                            changed_states[entity].append([attribute, other_attributes_map[state]])
                                        constructed_key  = entity + '_' + attribute
                                        if constructed_key in latest_states.keys() and other_attributes_map[state][0] != latest_states[constructed_key] and other_attributes_map[state][0] != 0: # latest state doesn't match pre-condition: indicates a conflict
                                            if entity not in confl_states.keys():
                                                confl_states[entity] = []
                                            confl_states[entity].append([attribute, other_attributes_map[state]])
                                        if other_attributes_map[state][1] != 0:
                                            latest_states[constructed_key] = other_attributes_map[state][1]
                                        states[entities.index(entity)][sentence_i][0][attributes.index(attribute)] = other_attributes_map[state][0]
                                        states[entities.index(entity)][sentence_i][1][attributes.index(attribute)] = other_attributes_map[state][1]
                        total_info_states.append(info_states)
                        total_changed_states.append(changed_states)
                        total_confl_states.append(confl_states)
                    story_json_data["states"] = states
                    story_json_data["info_states"] = total_info_states
                    story_json_data["changed_states"] = total_changed_states
                    story_json_data["confl_states"] = total_confl_states
                    stories.append(story_json_data)
                stories_json_data["stories"] = stories
                stories_json_data['example_id'] = object['example_id']
                if have_all_states:
                    if not(reduce_options) or (get_eff_attr_state_text(stories_json_data["stories"]) in all_options_reduced_aep and get_pre_attr_state_text(stories_json_data["stories"]) in all_options_reduced_app):
                        dataset.append(stories_json_data)
        else:
            raise ValueError("an example is missing labeled physical states")
    print("%s instances with multiple conflicts" % multi_conflicts)
    return dataset

def load_trip_dataset(exclude_multi_confl=False, condense_multiple_conflict=False, reduce_options=False):
    raw_all_data = json.load(open("../data/trip.json", "r"))
    raw_train = raw_all_data[0]["train"] # cloze
    raw_dev = raw_all_data[0]["dev"] # cloze
    raw_test = raw_all_data[0]["test"] # cloze
    trip_dataset = {}
    print("Loading train data...")
    trip_dataset["train"] = raw_dict_to_dataset(raw_train, exclude_multi_confl=exclude_multi_confl, condense_multiple_conflict=condense_multiple_conflict, reduce_options=reduce_options)
    print("Loading dev data...")
    trip_dataset["dev"] = raw_dict_to_dataset(raw_dev, exclude_multi_confl=exclude_multi_confl, condense_multiple_conflict=condense_multiple_conflict, reduce_options=reduce_options)
    print("Loading test data...")
    trip_dataset["test"] = raw_dict_to_dataset(raw_test, exclude_multi_confl=exclude_multi_confl, condense_multiple_conflict=condense_multiple_conflict, reduce_options=reduce_options)
    return trip_dataset

def is_human(entity):
    return (entity[0].isupper() and entity != 'TV')

attr_all = ["h_location", "conscious", "wearing", "h_wet", "hygiene", "location", "exist", "clean", "power", "functional", "pieces", "wet", "open", "temperature", "solid", "contain", "running", "moveable", "mixed", "edible"]
h_location_map = ['moved nowhere new', 'disappeared', 'moved somewhere new']
o_location_map = ['moved nowhere new', 'disappeared', 'picked up', 'put down', 'put on', 'removed', 'put into a container', 'taken out of a container', 'moved somewhere new']
other_attr_map = { 'conscious': ('unconscious', 'conscious'),
                    'wearing': ('undressed', 'dressed'), 
                    'h_wet': ('dry', 'wet'), 
                    'hygiene': ('dirty', 'clean'), 
                    'exist': ('no longer existent', 'existent'), 
                    'clean': ('dirty', 'clean'),
                    'power': ('unpowered', 'powered'), 
                    'functional': ('broken', 'functional'), 
                    'pieces': ('whole', 'in pieces'), 
                    'wet': ('dry', 'wet'), 
                    'open': ('closed', 'open'), 
                    'temperature': ('cold', 'hot'), 
                    'solid': ('fluid', 'solid'), 
                    'contain': ('empty', 'occupied'), 
                    'running': ('turned off', 'turned on'), 
                    'moveable': ('stuck', 'moveable'), 
                    'mixed': ('separated', 'mixed'), 
                    'edible': ('inedible', 'edible')}
other_attr_pre_effect_map = {0: [0, 0], 1:[1, 1], 2:[2, 2], 3:[2, 1], 4:[1, 2], 5:[0, 1], 6:[0, 2], 7:[1, 0], 8:[2, 0]}
attr_state_map = {}
for attr in other_attr_map.keys():
    if attr != "h_wet" and attr != "hygiene":
        attr_state_map[other_attr_map[attr][0]] = [attr_all.index(attr), 1]
        attr_state_map[other_attr_map[attr][1]] = [attr_all.index(attr), 2]
attr_state_map['dry_h'] = [3, 1]
attr_state_map['wet_h'] = [3, 2]
attr_state_map['dirty_h'] = [4, 1]
attr_state_map['clean_h'] = [4, 2]
attr_state_map['disappeared_h'] = [0, 1]
attr_state_map['moved somewhere new_h'] = [0, 2]
attr_state_map['disappeared'] = [5, 1]
attr_state_map['picked up'] = [5, 2]
attr_state_map['put down'] = [5, 3]
attr_state_map['put on'] = [5, 4]
attr_state_map['removed'] = [5, 5]
attr_state_map['put into a container'] = [5, 6]
attr_state_map['taken out of a container'] = [5, 7]
attr_state_map['moved somewhere new'] = [5, 8]
all_options = []

all_options = []
for attr in other_attr_map.keys():
    if attr != "h_wet" and attr != "hygiene":
        all_options.append(other_attr_map[attr][0])
        all_options.append(other_attr_map[attr][1])
for i in range (1, 9):
    loc = o_location_map[i]
    all_options.append(loc)
# all_options.append("irrelevant")

all_options_reduced_aep = ['no longer existent', 'broken', 'in pieces', 'turned off', 'inedible', 'unpowered']
all_options_reduced_app = ['existent', 'functional', 'whole', 'turned on', 'edible', 'powered']

# The original TRIP paper uses a heuristic to convert
# examples with more than one conflicting sentence pair to examples
# with only one conflicting sentence pair.
#
# (this may not be the best way to handle this issue, but was
# done just for simplicity)
def condense_multiple_conflict_pairs(conflict_sents, breakpoint):
    return [max([s for s in conflict_sents if s < breakpoint]), breakpoint]