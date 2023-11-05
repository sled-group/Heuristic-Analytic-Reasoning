import json
import re
from pprint import pprint
import numpy as np
import pickle
import os
from numpy.linalg import norm
import random

from data.trip import is_human, all_options, all_options_reduced_aep, all_options_reduced_app, attr_all, h_location_map, o_location_map, other_attr_map, attr_state_map
from utils import cosine_similarity, softmax, load_glove_vectors, glove_similarity_match


ATTR_DEFAULT_VALUES = [0, 2, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0]
glove = load_glove_vectors()

def balanced_sample_story_ids(all_stories, select_n_stories_for_each_category=2, enable_random_shuffle=False, random_seed=50):
    # IMPORTANT: if encounters any error, change a random seed!
    all_stories = list(enumerate(all_stories))
    random.seed(random_seed)
    if enable_random_shuffle: random.shuffle(all_stories)
    selected_idxes = []
    n_confl_entity_specific_stories = 0
    n_confl_entity_agnostic_stories = 0
    n_no_explicit_confl_stories = 0
    for idx, stories in all_stories:
        stories_info = stories['stories']
        if stories_info[0]['plausible'] == True:
            implausible_story = stories_info[1]
        else:
            implausible_story = stories_info[0]
        has_explicit_entity_agnostic_confl = False
        has_explicit_entity_specific_confl = False
        if entity_agnostic_confl_candidates_finder(implausible_story):
            has_explicit_entity_agnostic_confl = True
        if entity_specific_confl_candidates_finder(implausible_story):
            has_explicit_entity_specific_confl = True
        if n_confl_entity_specific_stories < select_n_stories_for_each_category and has_explicit_entity_specific_confl:
            selected_idxes.append(stories['example_id'])
            n_confl_entity_specific_stories += 1
        elif n_confl_entity_agnostic_stories < select_n_stories_for_each_category and has_explicit_entity_agnostic_confl:
            selected_idxes.append(stories['example_id'])
            n_confl_entity_agnostic_stories += 1
        elif n_no_explicit_confl_stories < select_n_stories_for_each_category and (not has_explicit_entity_specific_confl) and (not has_explicit_entity_agnostic_confl):
            selected_idxes.append(stories['example_id'])
            n_no_explicit_confl_stories += 1
        if n_confl_entity_specific_stories == select_n_stories_for_each_category and n_confl_entity_agnostic_stories == select_n_stories_for_each_category and n_no_explicit_confl_stories == select_n_stories_for_each_category:
            return selected_idxes
    return selected_idxes

def state_make_readable(sentence_idx, entity_idx, entities, states):
    this_states = {'sentence_idx': sentence_idx, 
                'entity_name': entities[entity_idx],
                'attribute_value_pairs': []}
    for attribute_idx, state_value in enumerate(states):
        if state_value > 0:
            attribute_name = attr_all[attribute_idx]
            if is_human(this_states['entity_name']):
                if attribute_name != 'location': # sanity check we don't get object location value
                    if attribute_name == 'h_location':
                        state_name = h_location_map[state_value]
                    else:
                        state_name = other_attr_map[attribute_name][state_value - 1]
                else:
                    # If we did, just skip it 
                    continue
            else:
                if attribute_name != 'h_location': # sanity check we don't get human location value
                    if attribute_name == 'location':
                        state_name = o_location_map[state_value]
                    else:
                        state_name = other_attr_map[attribute_name][state_value - 1]
                else:
                    # If we did, just skip it
                    continue
            this_states['attribute_value_pairs'].append((attribute_name, state_name))
    return this_states

def trip_metrics_for_one_pred(test_sample, pred):
    json_data = pred
    stories_info = test_sample['stories']
    implausible_story = None
    if stories_info[0]['plausible'] == True:
        implausible_story = stories_info[1]
    else:
        implausible_story = stories_info[0]
    # print("Line progress:", str(line_index))
    correct_plausibility = False

    if json_data["plausible_story"] == 'A':
        if stories_info[0]['plausible'] == True: 
            correct_plausibility = True
    else:
        if stories_info[1]['plausible'] == True: 
            correct_plausibility = True
    correct_confl_pairs = False
    if correct_plausibility:
        # if len(implausible_story["confl_pairs"]) > 1:
        #     print(implausible_story["confl_pairs"])
        if len(json_data["confl_pairs"]) > 0 and json_data["confl_pairs"][0] in implausible_story["confl_pairs"]:
            correct_confl_pairs = True
    correct_physical_states = False
    if correct_confl_pairs:
        # print(json_data['physical_states'])
        if len(json_data["physical_states"]) > 0:
            correct0, nondefault0 = state_predict(json_data["physical_states"][0][2], implausible_story["states"][json_data["physical_states"][0][0]][json_data["physical_states"][0][1]][1])
            correct1, nondefault1 = state_predict(json_data["physical_states"][1][2], implausible_story["states"][json_data["physical_states"][1][0]][json_data["physical_states"][1][1]][0])
        else:
            correct0, nondefault0 = False, False
            correct1, nondefault1 = False, False

        # Verifiability requires all predicted non-zero non-default states to be correct,
        # and at least one non-default state to be predicted within the conflict
        if correct0 and correct1 and (nondefault0 or nondefault1):
            correct_physical_states = True
    return correct_plausibility, correct_confl_pairs, correct_physical_states


def trip_metrics(test_dataset, preds):
    """Calculates the TRIP metrics and constructs readable model predictions/labels."""
    """entity_agnostic_confl_candidates_finder() will not detect conflict with the same entity name (entity-different)."""
    """Use entity_specific_confl_candidates_finder() or entity_agnostic_confl_candidates_finder() to make entity agnostic"""
    num_total_story_pairs_full = 0
    num_correct_plausibility_full = 0
    num_correct_confl_pairs_full = 0
    num_correct_physical_states_full = 0

    num_total_story_pairs_explicit_confl = 0
    num_correct_plausibility_explicit_confl = 0
    num_correct_confl_pairs_explicit_confl = 0
    num_correct_physical_states_explicit_confl = 0

    num_total_story_pairs_implicit_confl = 0
    num_correct_plausibility_implicit_confl = 0
    num_correct_confl_pairs_implicit_confl = 0
    num_correct_physical_states_implicit_confl = 0

    line_index = 0
    handle = preds
    for line in handle:
        json_data = line
        stories_info = test_dataset[line_index]['stories']
        implausible_story = None
        has_explicit_entity_agnostic_confl = False
        has_explicit_entity_specific_confl = False
        if stories_info[0]['plausible'] == True:
            implausible_story = stories_info[1]
        else:
            implausible_story = stories_info[0]
        if entity_specific_confl_candidates_finder(implausible_story):
            has_explicit_entity_specific_confl = True
        num_total_story_pairs_full += 1
        if has_explicit_entity_specific_confl:
            num_total_story_pairs_explicit_confl += 1
        if not has_explicit_entity_specific_confl:
            num_total_story_pairs_implicit_confl += 1
        # print("Line progress:", str(line_index))
        correct_plausibility = False

        if json_data["plausible_story"] == 'A':
            if stories_info[0]['plausible'] == True: 
                num_correct_plausibility_full += 1
                correct_plausibility = True
                if has_explicit_entity_specific_confl:
                    num_correct_plausibility_explicit_confl += 1
                if not has_explicit_entity_specific_confl:
                    num_correct_plausibility_implicit_confl += 1
        else:
            if stories_info[1]['plausible'] == True: 
                num_correct_plausibility_full += 1
                correct_plausibility = True
                if has_explicit_entity_specific_confl:
                    num_correct_plausibility_explicit_confl += 1
                if not has_explicit_entity_specific_confl:
                    num_correct_plausibility_implicit_confl += 1
        correct_confl_pairs = False
        if correct_plausibility:
            # if len(implausible_story["confl_pairs"]) > 1:
            #     print(implausible_story["confl_pairs"])
            if len(json_data["confl_pairs"]) > 0 and json_data["confl_pairs"][0] in implausible_story["confl_pairs"]:
                num_correct_confl_pairs_full += 1
                correct_confl_pairs = True
                if has_explicit_entity_specific_confl:
                    num_correct_confl_pairs_explicit_confl += 1
                if not has_explicit_entity_specific_confl:
                    num_correct_confl_pairs_implicit_confl += 1
        correct_physical_states = False
        if correct_confl_pairs:
            # print(json_data['physical_states'])
            if len(json_data["physical_states"]) > 0:
                correct0, nondefault0 = state_predict(json_data["physical_states"][0][2], implausible_story["states"][json_data["physical_states"][0][0]][json_data["physical_states"][0][1]][1])
                correct1, nondefault1 = state_predict(json_data["physical_states"][1][2], implausible_story["states"][json_data["physical_states"][1][0]][json_data["physical_states"][1][1]][0])
            else:
                correct0, nondefault0 = False, False
                correct1, nondefault1 = False, False

            # Verifiability requires all predicted non-zero non-default states to be correct,
            # and at least one non-default state to be predicted within the conflict
            if correct0 and correct1 and (nondefault0 or nondefault1):
                # print("generated text:", json_data['generated_text'])
                # print("extracted physical states:", physical_states_extractor_bottomup_compact(json_data['generated_text']))
                # print("ground truth effect:", state_make_readable(implausible_story["confl_pairs"][0][0], 
                #                                     json_data["physical_states"][0][0],
                #                                     implausible_story['entities'],
                #                                     implausible_story['states'][json_data["physical_states"][0][0]][implausible_story["confl_pairs"][0][0]][1]))
                # print("ground truth precondtition:", state_make_readable(implausible_story["confl_pairs"][0][1], 
                #                                     json_data["physical_states"][1][0], 
                #                                     implausible_story['entities'],
                #                                     implausible_story['states'][json_data["physical_states"][1][0]][implausible_story["confl_pairs"][0][1]][0]))
                # print('-' * 80)
                num_correct_physical_states_full += 1
                correct_physical_states = True
                if has_explicit_entity_specific_confl:
                    num_correct_physical_states_explicit_confl += 1
                if not has_explicit_entity_specific_confl:
                    num_correct_physical_states_implicit_confl += 1
        line_index += 1

        # Make physical states preds and labels more readable by converting labels to strings
        new_physical_states = []
        new_physical_states_gt = []
        if len(line['confl_pairs']) != 0:
            for entity_idx, sentence_idx, states in line['physical_states']:
                # Get predicted states
                this_states = state_make_readable(sentence_idx, 
                                                entity_idx, 
                                                implausible_story['entities'], 
                                                states)
                new_physical_states.append(this_states)
                
                # Get corresponding labels
                if sentence_idx == max(line['confl_pairs'][0]):
                    # If this is the second conflicting sentence, we check the precondition
                    precondition_effect_idx = 0
                elif sentence_idx == min(line['confl_pairs'][0]):
                    # If this is the first conflicting sentence, we check the effect
                    precondition_effect_idx = 1
                else:
                    raise ValueError("Mismatch of conflicting sentence indices in evaluation: %s, %s" % (str(sentence_idx), str(line['confl_pairs'])))

                this_states_gt = state_make_readable(sentence_idx, 
                                                    entity_idx, 
                                                    implausible_story['entities'],
                                                    implausible_story['states'][entity_idx][sentence_idx][precondition_effect_idx])
                new_physical_states_gt.append(this_states_gt)
        line['physical_states'] = new_physical_states

        # Save labels too
        line['plausible_story_gt'] = 0 if stories_info[0]['plausible'] else 1
        line['confl_pairs_gt'] = implausible_story['confl_pairs']
        line['physical_states_gt'] = new_physical_states_gt

        line['accurate'] = correct_plausibility
        line['consistent'] = correct_plausibility and correct_confl_pairs
        line['verifiable'] = correct_plausibility and correct_confl_pairs and correct_physical_states

    accuracy_plausibility_full = num_correct_plausibility_full / num_total_story_pairs_full
    accuracy_confl_pairs_full = num_correct_confl_pairs_full / num_total_story_pairs_full
    accuracy_physical_states_full = num_correct_physical_states_full / num_total_story_pairs_full

    if num_total_story_pairs_explicit_confl == 0:
        accuracy_plausibility_explicit_confl = None
        accuracy_confl_pairs_explicit_confl = None
        accuracy_physical_states_explicit_confl = None
    else:
        accuracy_plausibility_explicit_confl = num_correct_plausibility_explicit_confl / num_total_story_pairs_explicit_confl
        accuracy_confl_pairs_explicit_confl = num_correct_confl_pairs_explicit_confl / num_total_story_pairs_explicit_confl
        accuracy_physical_states_explicit_confl = num_correct_physical_states_explicit_confl / num_total_story_pairs_explicit_confl

    if num_total_story_pairs_implicit_confl == 0:
        accuracy_plausibility_implicit_confl = None
        accuracy_confl_pairs_implicit_confl = None
        accuracy_physical_states_implicit_confl = None
    else:
        accuracy_plausibility_implicit_confl = num_correct_plausibility_implicit_confl / num_total_story_pairs_implicit_confl
        accuracy_confl_pairs_implicit_confl = num_correct_confl_pairs_implicit_confl / num_total_story_pairs_implicit_confl
        accuracy_physical_states_implicit_confl = num_correct_physical_states_implicit_confl / num_total_story_pairs_implicit_confl

    return {
        "accuracy_full": accuracy_plausibility_full,
        "consistency_full": accuracy_confl_pairs_full,
        "verifiability_full": accuracy_physical_states_full,
        "accuracy_explicit_confl": accuracy_plausibility_explicit_confl,
        "consistency_explicit_confl": accuracy_confl_pairs_explicit_confl,
        "verifiability_explicit_confl": accuracy_physical_states_explicit_confl,
        "accuracy_implicit_confl": accuracy_plausibility_implicit_confl,
        "consistency_implicit_confl": accuracy_confl_pairs_implicit_confl,
        "verifiability_implicit_confl": accuracy_physical_states_implicit_confl,
    }, preds

# Consistent methods for prompt generation
def generate_text_options():
    text_options = "Physical state options: "
    for i in range(0, len(all_options) - 1):
        text_options = text_options + all_options[i] + ', '
    # text_options = text_options + "irrelevant"
    return text_options[:-2] + '\n'

def generate_text_options_reduced_aep():
    text_options = "Physical state options: "
    for i in range(0, len(all_options_reduced_aep)):
        text_options = text_options + all_options_reduced_aep[i] + ', '
    return text_options[:-2] + '\n'

def generate_text_options_reduced_app():
    text_options = "Physical state options: "
    for i in range(0, len(all_options_reduced_app)):
        text_options = text_options + all_options_reduced_app[i] + ', '
    return text_options[:-2] + '\n'

def app_prompt_generator(action_text, entity=None):
    if entity:
        if is_human(entity):
            entity_name = entity
            entity_name_bos = entity
        else:
            entity_name = "the " + entity
            entity_name_bos = "The " + entity
        if entity[-1] == 's':
            return f"Before {action_text}, what were the state of {entity_name}? {entity_name_bos} were "
        else:
            return f"Before {action_text}, what was the state of {entity_name}? {entity_name_bos} was "
    else:
        return f"Before {action_text}, what was the state of "
    
def app_prompt_generator_fully_separate(action_text, entity=None):
    if entity:
        if is_human(entity):
            entity_name = entity
            entity_name_bos = entity
        else:
            entity_name = "the " + entity
            entity_name_bos = "The " + entity
        if entity[-1] == 's':
            return f"Before, what were the state of {entity_name}? {entity_name_bos} were "
        else:
            return f"Before, what was the state of {entity_name}? {entity_name_bos} was "
    else:
        return f"Before, what was the state of "
    
def app_prompt_generator_fully_separate_familiarization(action_text, entity=None):
    if entity:
        if is_human(entity):
            entity_name = entity
            entity_name_bos = entity
        else:
            entity_name = "the " + entity
            entity_name_bos = "The " + entity
        if entity[-1] == 's':
            return f"{action_text}. Before, what were the state of {entity_name}? {entity_name_bos} were "
        else:
            return f"{action_text}. Before, what was the state of {entity_name}? {entity_name_bos} was "
    else:
        return f"{action_text}. Before, what was the state of "

def aep_prompt_generator(action_text, entity=None):
    if entity:
        if is_human(entity):
            entity_name = entity
            entity_name_bos = entity
        else:
            entity_name = "the " + entity
            entity_name_bos = "The " + entity
        if entity[-1] == 's':
            return f"After {action_text}, what are the state of {entity_name}? {entity_name_bos} are now "
        else:
            return f"After {action_text}, what is the state of {entity_name}? {entity_name_bos} is now "
    else:
        return f"After {action_text}, what is the state of "
    
def aep_prompt_generator_fully_separate(action_text, entity=None):
    if entity:
        if is_human(entity):
            entity_name = entity
            entity_name_bos = entity
        else:
            entity_name = "the " + entity
            entity_name_bos = "The " + entity
        if entity[-1] == 's':
            return f"After, what are the state of {entity_name}? {entity_name_bos} are now "
        else:
            return f"After, what is the state of {entity_name}? {entity_name_bos} is now "
    else:
        return f"After, what is the state of "
    
def aep_prompt_generator_fully_separate_familiarization(action_text, entity=None):
    if entity:
        if is_human(entity):
            entity_name = entity
            entity_name_bos = entity
        else:
            entity_name = "the " + entity
            entity_name_bos = "The " + entity
        if entity[-1] == 's':
            return f"{action_text}. After, what are the state of {entity_name}? {entity_name_bos} are now "
        else:
            return f"{action_text}. After, what is the state of {entity_name}? {entity_name_bos} is now "
    else:
        return f"{action_text}. After, what is the state of "

def app_demo_generator(action_text, entity, state_text):
    return app_prompt_generator(action_text, entity=entity) + state_text + ".\n"

def aep_demo_generator(action_text, entity, state_text):
    return aep_prompt_generator(action_text, entity=entity) + state_text + ".\n"

def app_demo_generator_fully_separate(action_text, entity, state_text):
    return app_prompt_generator_fully_separate(action_text, entity=entity) + state_text + ".\n"

def aep_demo_generator_fully_separate(action_text, entity, state_text):
    return aep_prompt_generator_fully_separate(action_text, entity=entity) + state_text + ".\n"

def app_demo_generator_fully_separate_familiarization(action_text, entity, state_text):
    return app_prompt_generator_fully_separate_familiarization(action_text, entity=entity) + state_text + ".\n"

def aep_demo_generator_fully_separate_familiarization(action_text, entity, state_text):
    return aep_prompt_generator_fully_separate_familiarization(action_text, entity=entity) + state_text + ".\n"

def conflict_prompt_generator(implausible_story_letter=None):
    """Generate a prompt for conflict information about a story from a pair of stories."""
    if implausible_story_letter:
        return f"In Story {implausible_story_letter}, "
    else:
        return ""

def conflict_demo_generator(sentence_numbers, implausible_story_letter=None, conflict_reason=None):
    """Generate demo for conflict information about a story."""
    assert len(sentence_numbers) == 2
    if conflict_reason:
        return f"{conflict_prompt_generator(implausible_story_letter)}{conflict_reason}. Therefore, sentences {sentence_numbers[0]} and {sentence_numbers[1]} conflict with each other.\n"
    else:
        return f"{conflict_prompt_generator(implausible_story_letter)}sentences {sentence_numbers[0]} and {sentence_numbers[1]} conflict with each other.\n"
    
def conflict_demo_generator_bottom_up(sentence_numbers, implausible_story_letter, conflict_reason=None):
    """Generate demo for conflict information about a story."""
    assert len(sentence_numbers) == 2
    if conflict_reason:
        return f"{conflict_reason}. Therefore, sentences {sentence_numbers[0]} and {sentence_numbers[1]} conflict with each other.\n"
    else:
        return f"Therefore, sentences {sentence_numbers[0]} and {sentence_numbers[1]} conflict with each other.\n"
    
def conflict_demo_generator_fully_sep(sentence_numbers, implausible_story_letter, conflict_reason=None):
    """Generate demo for conflict information about a story."""
    assert len(sentence_numbers) == 2
    if conflict_reason:
        return f"{conflict_reason}. Sentences {sentence_numbers[0]} and {sentence_numbers[1]} conflict with each other in story {implausible_story_letter}.\n"
    else:
        return f"Sentences {sentence_numbers[0]} and {sentence_numbers[1]} conflict with each other in story {implausible_story_letter}.\n"
    
def story_prompt_generator(story):
    """Generate a prompt for information about a single story."""
    prompt = "Story: " + '\n'
    for sentence_i in range(0, len(story['sentences'])):
        sentence = story['sentences'][sentence_i]
        prompt += str(sentence_i + 1) + ". " + sentence + '\n'
    return prompt

def story_pair_prompt_generator(story_A, story_B):
    """Generate a prompt for information about a pair of stories."""
    prompt = "Story A: " + '\n'
    for sentence_i in range(0, len(story_A['sentences'])):
        sentence = story_A['sentences'][sentence_i]
        prompt = prompt + str(sentence_i + 1) + ". " + sentence + '\n'
    prompt = prompt + "Story B: " + '\n'
    for sentence_i in range(0, len(story_B['sentences'])):
        sentence = story_B['sentences'][sentence_i]
        prompt = prompt + str(sentence_i + 1) + ". " + sentence + '\n'
    return prompt

def plausibility_demo_generator(plausible_story_letter):
    """Generate demo for plausibility information about a story pair."""
    return f"Story {plausible_story_letter} is more plausible.\n"

def plausibility_demo_generator_ask_implausible(implausible_story_letter):
    """Generate demo for plausibility information about a story pair."""
    return f"Story {implausible_story_letter} is more implausible.\n"

def plausibility_demo_generator_bottomup(plausible_story_letter):
    """Generate demo for plausibility information about a story pair."""
    return f"Therefore, story {plausible_story_letter} is more plausible.\n"

def entity_specific_confl_candidates_finder(implausible_story):
    """Find candidates with a mismatch on effect state and precondition state specific to the entity name"""
    """If the returned list is empty, there's no explicit conflict from physical states (on the same attribute)"""
    candidates = []
    confl_sent_index_1 = implausible_story["confl_pairs"][0][0]
    confl_sent_index_2 = implausible_story["confl_pairs"][0][1]
    for entity_idx, entity in enumerate(implausible_story['entities']):
        effect_states = implausible_story['states'][entity_idx][confl_sent_index_1][1]
        precondition_states = implausible_story['states'][entity_idx][confl_sent_index_2][0]
        for state_i in range(0, 20):
            attribute = attr_all[state_i]
            eff_state = effect_states[state_i]
            pre_state = precondition_states[state_i]
            if (eff_state == 1 and pre_state == 2) or (eff_state == 2 and pre_state == 1):
                candidates.append({"entity": entity, "attribute": attribute, "eff_state": eff_state, "pre_state": pre_state})
    return candidates

def entity_agnostic_confl_candidates_finder(implausible_story):
    """Find candidates with a mismatch on effect state and precondition state regardless of the entity name"""
    """If the returned dictionary is empty, there's no explicit conflict from physical states (on the same attribute)"""
    """This method will not detect conflict with the same entity name (entity-different)."""
    """Use entity_specific_confl_candidates_finder and this method together to make entity agnostic"""
    dict_attribute_entity_states = {}
    confl_sent_index_1 = implausible_story["confl_pairs"][0][0]
    confl_sent_index_2 = implausible_story["confl_pairs"][0][1]
    for entity_idx, entity in enumerate(implausible_story['entities']):
        effect_states = implausible_story['states'][entity_idx][confl_sent_index_1][1]
        precondition_states = implausible_story['states'][entity_idx][confl_sent_index_2][0]
        for state_i in range(0, 20):
            attribute = attr_all[state_i]
            eff_state = effect_states[state_i]
            pre_state = precondition_states[state_i]
            if attribute in dict_attribute_entity_states.keys():
                dict_attribute_entity_states[attribute].append({"entity": entity, "eff_state": eff_state, "pre_state": pre_state})
            else:
                dict_attribute_entity_states[attribute] = []
                dict_attribute_entity_states[attribute].append({"entity": entity, "eff_state": eff_state, "pre_state": pre_state})
    candidates = {}
    for attribute in dict_attribute_entity_states.keys():
        filtered_list = []
        entity_states_list = dict_attribute_entity_states[attribute]
        for i in range(0, len(entity_states_list)):
            for j in range(i + 1, len(entity_states_list)):
                first_entity_states, second_entity_states = entity_states_list[i], entity_states_list[j]
                if first_entity_states["eff_state"] != 0 and second_entity_states["pre_state"] != 0 and first_entity_states["eff_state"] != second_entity_states["pre_state"]:
                    filtered_list.append([first_entity_states, second_entity_states])
        if len(filtered_list) > 0:
            candidates[attribute] = filtered_list
    return candidates

def implausible_story_conflict_reason_generator(implausible_story):
    confl_sent1 = implausible_story["sentences"][implausible_story["confl_pairs"][0][0]]
    confl_sent2 = implausible_story["sentences"][implausible_story["confl_pairs"][0][1]]
    return f"{confl_sent1[:-1]}, but then {confl_sent2[:-1]}"

def physical_states_demo_generator_fully_separate(implausible_story):
    confl_sent_index_1 = implausible_story["confl_pairs"][0][0]
    confl_sent_index_2 = implausible_story["confl_pairs"][0][1]
    confl_sent_1 = implausible_story['sentences'][confl_sent_index_1]
    confl_sent_2 = implausible_story['sentences'][confl_sent_index_2]
    entity_specific_candidates = entity_specific_confl_candidates_finder(implausible_story)
    max_cosine_similarity = 0
    chosen_candidate = None
    if len(entity_specific_candidates) == 0:
        entity_agnostic_candidates = entity_agnostic_confl_candidates_finder(implausible_story)
        for attribute in entity_agnostic_candidates.keys():
            for pair in entity_agnostic_candidates[attribute]:
                tokens_1 = pair[0]["entity"].split()
                tokens_2 = pair[1]["entity"].split()
                sum_tokens_1_embed = np.zeros(50)
                sum_tokens_2_embed = np.zeros(50)
                for token in tokens_1:
                    sum_tokens_1_embed += np.array(glove[token])
                for token in tokens_2:
                    sum_tokens_2_embed += np.array(glove[token])
                embedding_1 = sum_tokens_1_embed/len(tokens_1) #avg
                embedding_2 = sum_tokens_2_embed/len(tokens_2) #avg
                # print(embedding_1, embedding_2)
                cos_sim = np.dot(embedding_1,embedding_2)/(norm(embedding_1)*norm(embedding_2))
                if cos_sim > max_cosine_similarity:
                    max_cosine_similarity = cos_sim
                    chosen_candidate = {
                        "entity_1": pair[0]["entity"], 
                        "entity_2": pair[1]["entity"], 
                        "attribute": attribute, 
                        "eff_state": pair[0]["eff_state"], 
                        "pre_state": pair[1]["pre_state"]
                    }
        if chosen_candidate:
            entity_1 = chosen_candidate["entity_1"]
            entity_2 = chosen_candidate["entity_2"]
        else:
            pre_state_info_list = []
            eff_state_info_list = []
            for entity_idx, entity in enumerate(implausible_story['entities']):
                effect_states = implausible_story['states'][entity_idx][confl_sent_index_1][1]
                precondition_states = implausible_story['states'][entity_idx][confl_sent_index_2][0]
                for state_i in range(0, 20):
                    attribute = attr_all[state_i]
                    pre_state = precondition_states[state_i]
                    eff_state = effect_states[state_i]
                    if eff_state != 0 and eff_state != ATTR_DEFAULT_VALUES[state_i]:
                        eff_state_info_list.append({"entity": entity, "attribute": attribute, "eff_state": eff_state})
                    if pre_state != 0 and pre_state != ATTR_DEFAULT_VALUES[state_i]:
                        pre_state_info_list.append({"entity": entity, "attribute": attribute, "pre_state": pre_state})
            random.seed(100)
            random.shuffle(pre_state_info_list)
            random.seed(200)
            random.shuffle(eff_state_info_list)
            pre_state_info = pre_state_info_list[0]
            eff_state_info = eff_state_info_list[0]
            chosen_candidate = {
                "entity_1": eff_state_info["entity"], 
                "entity_2": pre_state_info["entity"], 
                "attribute_1": eff_state_info["attribute"], 
                "attribute_2": pre_state_info["attribute"],
                "eff_state": eff_state_info["eff_state"], 
                "pre_state": pre_state_info["pre_state"]
            }
            entity_1 = chosen_candidate["entity_1"]
            entity_2 = chosen_candidate["entity_2"]
    else:
        chosen_candidate = entity_specific_candidates[0]
        entity_1 = chosen_candidate["entity"]
        entity_2 = chosen_candidate["entity"]
    if "attribute_1" in chosen_candidate.keys() and "attribute_2" in chosen_candidate.keys():
        attribute_1 = chosen_candidate["attribute_1"]
        attribute_2 = chosen_candidate["attribute_2"]
    else:
        attribute_1 = chosen_candidate["attribute"]
        attribute_2 = chosen_candidate["attribute"]
    eff_state = chosen_candidate["eff_state"]
    pre_state = chosen_candidate["pre_state"]
    if attribute_1 == "h_location":
        eff_attr_state_text = h_location_map[eff_state]
    elif attribute_1 == "location":
        eff_attr_state_text = o_location_map[eff_state]
    else:
        eff_attr_state_text = other_attr_map[attribute_1][eff_state - 1]
    if attribute_2 == "h_location":
        pre_attr_state_text = h_location_map[pre_state]
    elif attribute_2 == "location":
        pre_attr_state_text = o_location_map[pre_state]
    else:
        pre_attr_state_text = other_attr_map[attribute_2][pre_state - 1]
    return aep_demo_generator_fully_separate(confl_sent_1[:-1], entity_1, eff_attr_state_text) + app_demo_generator_fully_separate(confl_sent_2[:-1], entity_2, pre_attr_state_text)

def story_pair_demo_generator_topdown_ask_implausible(story_pair, conflict_reason_enabled=False, reasoning_depth='verifiable'):
    """Generates demo for TRIP two-story task in top-down mode."""
    story_A = story_pair[0]
    story_B = story_pair[1]
    demonstration = story_pair_prompt_generator(story_A, story_B)
    # demonstration = ""
    
    # Add story plausibility prediction
    if story_A["plausible"] == True:
        demonstration += plausibility_demo_generator_ask_implausible("B")
        implausible_story = story_B
        implausible_story_letter = "B"
    elif story_B["plausible"] == True:
        demonstration += plausibility_demo_generator_ask_implausible("A")
        implausible_story = story_A
        implausible_story_letter = "A"
    
    # Add conflicting sentence prediction and explanation
    if reasoning_depth in ["consistent", "verifiable"]:
        sentence_numbers = (implausible_story['confl_pairs'][0][0] + 1, implausible_story['confl_pairs'][0][1] + 1)
        if conflict_reason_enabled:
            demonstration += conflict_demo_generator(sentence_numbers,
                                                 implausible_story_letter=implausible_story_letter,
                                                 conflict_reason=implausible_story_conflict_reason_generator(implausible_story))
        else:
            demonstration += conflict_demo_generator(sentence_numbers,
                                                 implausible_story_letter=implausible_story_letter,
                                                 conflict_reason=None)

    # Add explanation of physical states in conflicting sentences
    if reasoning_depth == "verifiable":
        confl_sent_index_1 = implausible_story["confl_pairs"][0][0]
        confl_sent_index_2 = implausible_story["confl_pairs"][0][1]
        confl_sent_1 = implausible_story['sentences'][confl_sent_index_1]
        confl_sent_2 = implausible_story['sentences'][confl_sent_index_2]
        entity_specific_candidates = entity_specific_confl_candidates_finder(implausible_story)
        max_cosine_similarity = 0
        chosen_candidate = None
        if len(entity_specific_candidates) == 0:
            entity_agnostic_candidates = entity_agnostic_confl_candidates_finder(implausible_story)
            for attribute in entity_agnostic_candidates.keys():
                for pair in entity_agnostic_candidates[attribute]:
                    tokens_1 = pair[0]["entity"].split()
                    tokens_2 = pair[1]["entity"].split()
                    sum_tokens_1_embed = np.zeros(50)
                    sum_tokens_2_embed = np.zeros(50)
                    for token in tokens_1:
                        sum_tokens_1_embed += np.array(glove[token])
                    for token in tokens_2:
                        sum_tokens_2_embed += np.array(glove[token])
                    embedding_1 = sum_tokens_1_embed/len(tokens_1) #avg
                    embedding_2 = sum_tokens_2_embed/len(tokens_2) #avg
                    # print(embedding_1, embedding_2)
                    cos_sim = np.dot(embedding_1,embedding_2)/(norm(embedding_1)*norm(embedding_2))
                    if cos_sim > max_cosine_similarity:
                        max_cosine_similarity = cos_sim
                        chosen_candidate = {
                            "entity_1": pair[0]["entity"], 
                            "entity_2": pair[1]["entity"], 
                            "attribute": attribute, 
                            "eff_state": pair[0]["eff_state"], 
                            "pre_state": pair[1]["pre_state"]
                        }
            if chosen_candidate:
                entity_1 = chosen_candidate["entity_1"]
                entity_2 = chosen_candidate["entity_2"]
            else:
                pre_state_info_list = []
                eff_state_info_list = []
                # Sometimes there will be no interesting physical states in either the precondition or effect,
                # so we can also have a backup of default states
                pre_state_info_list_backup = []
                eff_state_info_list_backup = []
                for entity_idx, entity in enumerate(implausible_story['entities']):
                    effect_states = implausible_story['states'][entity_idx][confl_sent_index_1][1]
                    precondition_states = implausible_story['states'][entity_idx][confl_sent_index_2][0]
                    for state_i in range(0, 20):
                        attribute = attr_all[state_i]
                        pre_state = precondition_states[state_i]
                        eff_state = effect_states[state_i]
                        if eff_state != 0 and eff_state != ATTR_DEFAULT_VALUES[state_i]:
                            eff_state_info_list.append({"entity": entity, "attribute": attribute, "eff_state": eff_state})
                        if eff_state != ATTR_DEFAULT_VALUES[state_i]:
                            eff_state_info_list_backup.append({"entity": entity, "attribute": attribute, "eff_state": eff_state})
                        if pre_state != 0 and pre_state != ATTR_DEFAULT_VALUES[state_i]:
                            pre_state_info_list.append({"entity": entity, "attribute": attribute, "pre_state": pre_state})
                        if pre_state != ATTR_DEFAULT_VALUES[state_i]:
                            pre_state_info_list_backup.append({"entity": entity, "attribute": attribute, "pre_state": pre_state})
                random.seed(100)
                random.shuffle(pre_state_info_list)
                random.shuffle(pre_state_info_list_backup)
                random.seed(200)
                random.shuffle(eff_state_info_list)
                random.shuffle(eff_state_info_list_backup)
                if len(pre_state_info_list) > 0:
                    pre_state_info = pre_state_info_list[0]
                else:
                    pre_state_info = pre_state_info_list_backup[0]
                if len(eff_state_info_list):
                    eff_state_info = eff_state_info_list[0]
                else:
                    eff_state_info = eff_state_info_list_backup[0]
                chosen_candidate = {
                    "entity_1": eff_state_info["entity"], 
                    "entity_2": pre_state_info["entity"], 
                    "attribute_1": eff_state_info["attribute"], 
                    "attribute_2": pre_state_info["attribute"],
                    "eff_state": eff_state_info["eff_state"], 
                    "pre_state": pre_state_info["pre_state"]
                }
                entity_1 = chosen_candidate["entity_1"]
                entity_2 = chosen_candidate["entity_2"]
        else:
            chosen_candidate = entity_specific_candidates[0]
            entity_1 = chosen_candidate["entity"]
            entity_2 = chosen_candidate["entity"]
        if "attribute_1" in chosen_candidate.keys() and "attribute_2" in chosen_candidate.keys():
            attribute_1 = chosen_candidate["attribute_1"]
            attribute_2 = chosen_candidate["attribute_2"]
        else:
            attribute_1 = chosen_candidate["attribute"]
            attribute_2 = chosen_candidate["attribute"]
        eff_state = chosen_candidate["eff_state"]
        pre_state = chosen_candidate["pre_state"]
        if attribute_1 == "h_location":
            eff_attr_state_text = h_location_map[eff_state]
        elif attribute_1 == "location":
            eff_attr_state_text = o_location_map[eff_state]
        else:
            eff_attr_state_text = other_attr_map[attribute_1][eff_state - 1]
        if attribute_2 == "h_location":
            pre_attr_state_text = h_location_map[pre_state]
        elif attribute_2 == "location":
            pre_attr_state_text = o_location_map[pre_state]
        else:
            pre_attr_state_text = other_attr_map[attribute_2][pre_state - 1]
        demonstration += "For sentence " + str(confl_sent_index_1 + 1) + ':\n' + aep_demo_generator(confl_sent_1[:-1], entity_1, eff_attr_state_text)
        demonstration += "For sentence " + str(confl_sent_index_2 + 1) + ':\n' + app_demo_generator(confl_sent_2[:-1], entity_2, pre_attr_state_text)

    return demonstration

def story_pair_demo_generator_topdown(story_pair, conflict_reason_enabled=False, reasoning_depth='verifiable'):
    """Generates demo for TRIP two-story task in top-down mode."""
    story_A = story_pair[0]
    story_B = story_pair[1]
    demonstration = story_pair_prompt_generator(story_A, story_B)
    # demonstration = ""
    
    # Add story plausibility prediction
    if story_A["plausible"] == True:
        demonstration += plausibility_demo_generator("A")
        implausible_story = story_B
        implausible_story_letter = "B"
    elif story_B["plausible"] == True:
        demonstration += plausibility_demo_generator("B")
        implausible_story = story_A
        implausible_story_letter = "A"
    
    # Add conflicting sentence prediction and explanation
    if reasoning_depth in ["consistent", "verifiable"]:
        sentence_numbers = (implausible_story['confl_pairs'][0][0] + 1, implausible_story['confl_pairs'][0][1] + 1)
        if conflict_reason_enabled:
            demonstration += conflict_demo_generator(sentence_numbers,
                                                 implausible_story_letter=implausible_story_letter,
                                                 conflict_reason=implausible_story_conflict_reason_generator(implausible_story))
        else:
            demonstration += conflict_demo_generator(sentence_numbers,
                                                 implausible_story_letter=implausible_story_letter,
                                                 conflict_reason=None)

    # Add explanation of physical states in conflicting sentences
    if reasoning_depth == "verifiable":
        confl_sent_index_1 = implausible_story["confl_pairs"][0][0]
        confl_sent_index_2 = implausible_story["confl_pairs"][0][1]
        confl_sent_1 = implausible_story['sentences'][confl_sent_index_1]
        confl_sent_2 = implausible_story['sentences'][confl_sent_index_2]
        entity_specific_candidates = entity_specific_confl_candidates_finder(implausible_story)
        max_cosine_similarity = 0
        chosen_candidate = None
        if len(entity_specific_candidates) == 0:
            entity_agnostic_candidates = entity_agnostic_confl_candidates_finder(implausible_story)
            for attribute in entity_agnostic_candidates.keys():
                for pair in entity_agnostic_candidates[attribute]:
                    tokens_1 = pair[0]["entity"].split()
                    tokens_2 = pair[1]["entity"].split()
                    sum_tokens_1_embed = np.zeros(50)
                    sum_tokens_2_embed = np.zeros(50)
                    for token in tokens_1:
                        sum_tokens_1_embed += np.array(glove[token])
                    for token in tokens_2:
                        sum_tokens_2_embed += np.array(glove[token])
                    embedding_1 = sum_tokens_1_embed/len(tokens_1) #avg
                    embedding_2 = sum_tokens_2_embed/len(tokens_2) #avg
                    # print(embedding_1, embedding_2)
                    cos_sim = np.dot(embedding_1,embedding_2)/(norm(embedding_1)*norm(embedding_2))
                    if cos_sim > max_cosine_similarity:
                        max_cosine_similarity = cos_sim
                        chosen_candidate = {
                            "entity_1": pair[0]["entity"], 
                            "entity_2": pair[1]["entity"], 
                            "attribute": attribute, 
                            "eff_state": pair[0]["eff_state"], 
                            "pre_state": pair[1]["pre_state"]
                        }
            if chosen_candidate:
                entity_1 = chosen_candidate["entity_1"]
                entity_2 = chosen_candidate["entity_2"]
            else:
                pre_state_info_list = []
                eff_state_info_list = []
                # Sometimes there will be no interesting physical states in either the precondition or effect,
                # so we can also have a backup of default states
                pre_state_info_list_backup = []
                eff_state_info_list_backup = []
                for entity_idx, entity in enumerate(implausible_story['entities']):
                    effect_states = implausible_story['states'][entity_idx][confl_sent_index_1][1]
                    precondition_states = implausible_story['states'][entity_idx][confl_sent_index_2][0]
                    for state_i in range(0, 20):
                        attribute = attr_all[state_i]
                        pre_state = precondition_states[state_i]
                        eff_state = effect_states[state_i]
                        if eff_state != 0 and eff_state != ATTR_DEFAULT_VALUES[state_i]:
                            eff_state_info_list.append({"entity": entity, "attribute": attribute, "eff_state": eff_state})
                        if eff_state != ATTR_DEFAULT_VALUES[state_i]:
                            eff_state_info_list_backup.append({"entity": entity, "attribute": attribute, "eff_state": eff_state})
                        if pre_state != 0 and pre_state != ATTR_DEFAULT_VALUES[state_i]:
                            pre_state_info_list.append({"entity": entity, "attribute": attribute, "pre_state": pre_state})
                        if pre_state != ATTR_DEFAULT_VALUES[state_i]:
                            pre_state_info_list_backup.append({"entity": entity, "attribute": attribute, "pre_state": pre_state})
                random.seed(100)
                random.shuffle(pre_state_info_list)
                random.shuffle(pre_state_info_list_backup)
                random.seed(200)
                random.shuffle(eff_state_info_list)
                random.shuffle(eff_state_info_list_backup)
                if len(pre_state_info_list) > 0:
                    pre_state_info = pre_state_info_list[0]
                else:
                    pre_state_info = pre_state_info_list_backup[0]
                if len(eff_state_info_list):
                    eff_state_info = eff_state_info_list[0]
                else:
                    eff_state_info = eff_state_info_list_backup[0]
                chosen_candidate = {
                    "entity_1": eff_state_info["entity"], 
                    "entity_2": pre_state_info["entity"], 
                    "attribute_1": eff_state_info["attribute"], 
                    "attribute_2": pre_state_info["attribute"],
                    "eff_state": eff_state_info["eff_state"], 
                    "pre_state": pre_state_info["pre_state"]
                }
                entity_1 = chosen_candidate["entity_1"]
                entity_2 = chosen_candidate["entity_2"]
        else:
            chosen_candidate = entity_specific_candidates[0]
            entity_1 = chosen_candidate["entity"]
            entity_2 = chosen_candidate["entity"]
        if "attribute_1" in chosen_candidate.keys() and "attribute_2" in chosen_candidate.keys():
            attribute_1 = chosen_candidate["attribute_1"]
            attribute_2 = chosen_candidate["attribute_2"]
        else:
            attribute_1 = chosen_candidate["attribute"]
            attribute_2 = chosen_candidate["attribute"]
        eff_state = chosen_candidate["eff_state"]
        pre_state = chosen_candidate["pre_state"]
        if attribute_1 == "h_location":
            eff_attr_state_text = h_location_map[eff_state]
        elif attribute_1 == "location":
            eff_attr_state_text = o_location_map[eff_state]
        else:
            eff_attr_state_text = other_attr_map[attribute_1][eff_state - 1]
        if attribute_2 == "h_location":
            pre_attr_state_text = h_location_map[pre_state]
        elif attribute_2 == "location":
            pre_attr_state_text = o_location_map[pre_state]
        else:
            pre_attr_state_text = other_attr_map[attribute_2][pre_state - 1]
        demonstration += "For sentence " + str(confl_sent_index_1 + 1) + ':\n' + aep_demo_generator(confl_sent_1[:-1], entity_1, eff_attr_state_text)
        demonstration += "For sentence " + str(confl_sent_index_2 + 1) + ':\n' + app_demo_generator(confl_sent_2[:-1], entity_2, pre_attr_state_text)

    return demonstration

def story_pair_demo_generator_bottomup_compact(story_pair, conflict_reason_enabled=False, reasoning_depth='verifiable'):
    """Generates demo for TRIP two-story task in bottom-up compact mode."""
    story_A = story_pair[0]
    story_B = story_pair[1]
    demonstration = story_pair_prompt_generator(story_A, story_B)
    # demonstration = ""

    if story_A["plausible"] == True:
        implausible_story = story_B
        implausible_story_letter = "B"
    elif story_B["plausible"] == True:
        implausible_story = story_A
        implausible_story_letter = "A"

    # Add explanation of physical states in conflicting sentences
    if reasoning_depth == "verifiable":
        confl_sent_index_1 = implausible_story["confl_pairs"][0][0]
        confl_sent_index_2 = implausible_story["confl_pairs"][0][1]
        confl_sent_1 = implausible_story['sentences'][confl_sent_index_1]
        confl_sent_2 = implausible_story['sentences'][confl_sent_index_2]
        entity_specific_candidates = entity_specific_confl_candidates_finder(implausible_story)
        max_cosine_similarity = 0
        chosen_candidate = None
        if len(entity_specific_candidates) == 0:
            entity_agnostic_candidates = entity_agnostic_confl_candidates_finder(implausible_story)
            for attribute in entity_agnostic_candidates.keys():
                for pair in entity_agnostic_candidates[attribute]:
                    tokens_1 = pair[0]["entity"].split()
                    tokens_2 = pair[1]["entity"].split()
                    sum_tokens_1_embed = np.zeros(50)
                    sum_tokens_2_embed = np.zeros(50)
                    for token in tokens_1:
                        sum_tokens_1_embed += np.array(glove[token])
                    for token in tokens_2:
                        sum_tokens_2_embed += np.array(glove[token])
                    embedding_1 = sum_tokens_1_embed/len(tokens_1) #avg
                    embedding_2 = sum_tokens_2_embed/len(tokens_2) #avg
                    # print(embedding_1, embedding_2)
                    cos_sim = np.dot(embedding_1,embedding_2)/(norm(embedding_1)*norm(embedding_2))
                    if cos_sim > max_cosine_similarity:
                        max_cosine_similarity = cos_sim
                        chosen_candidate = {
                            "entity_1": pair[0]["entity"], 
                            "entity_2": pair[1]["entity"], 
                            "attribute": attribute, 
                            "eff_state": pair[0]["eff_state"], 
                            "pre_state": pair[1]["pre_state"]
                        }
            if chosen_candidate:
                entity_1 = chosen_candidate["entity_1"]
                entity_2 = chosen_candidate["entity_2"]
            else:
                pre_state_info_list = []
                eff_state_info_list = []
                for entity_idx, entity in enumerate(implausible_story['entities']):
                    effect_states = implausible_story['states'][entity_idx][confl_sent_index_1][1]
                    precondition_states = implausible_story['states'][entity_idx][confl_sent_index_2][0]
                    for state_i in range(0, 20):
                        attribute = attr_all[state_i]
                        pre_state = precondition_states[state_i]
                        eff_state = effect_states[state_i]
                        if eff_state != 0 and eff_state != ATTR_DEFAULT_VALUES[state_i]:
                            eff_state_info_list.append({"entity": entity, "attribute": attribute, "eff_state": eff_state})
                        if pre_state != 0 and pre_state != ATTR_DEFAULT_VALUES[state_i]:
                            pre_state_info_list.append({"entity": entity, "attribute": attribute, "pre_state": pre_state})
                random.seed(100)
                random.shuffle(pre_state_info_list)
                random.seed(200)
                random.shuffle(eff_state_info_list)
                pre_state_info = pre_state_info_list[0]
                eff_state_info = eff_state_info_list[0]
                chosen_candidate = {
                    "entity_1": eff_state_info["entity"], 
                    "entity_2": pre_state_info["entity"], 
                    "attribute_1": eff_state_info["attribute"], 
                    "attribute_2": pre_state_info["attribute"],
                    "eff_state": eff_state_info["eff_state"], 
                    "pre_state": pre_state_info["pre_state"]
                }
                entity_1 = chosen_candidate["entity_1"]
                entity_2 = chosen_candidate["entity_2"]
        else:
            chosen_candidate = entity_specific_candidates[0]
            entity_1 = chosen_candidate["entity"]
            entity_2 = chosen_candidate["entity"]
        if "attribute_1" in chosen_candidate.keys() and "attribute_2" in chosen_candidate.keys():
            attribute_1 = chosen_candidate["attribute_1"]
            attribute_2 = chosen_candidate["attribute_2"]
        else:
            attribute_1 = chosen_candidate["attribute"]
            attribute_2 = chosen_candidate["attribute"]
        eff_state = chosen_candidate["eff_state"]
        pre_state = chosen_candidate["pre_state"]
        if attribute_1 == "h_location":
            eff_attr_state_text = h_location_map[eff_state]
        elif attribute_1 == "location":
            eff_attr_state_text = o_location_map[eff_state]
        else:
            eff_attr_state_text = other_attr_map[attribute_1][eff_state - 1]
        if attribute_2 == "h_location":
            pre_attr_state_text = h_location_map[pre_state]
        elif attribute_2 == "location":
            pre_attr_state_text = o_location_map[pre_state]
        else:
            pre_attr_state_text = other_attr_map[attribute_2][pre_state - 1]
        demonstration += aep_demo_generator_fully_separate(confl_sent_1[:-1], entity_1, eff_attr_state_text)
        demonstration += app_demo_generator_fully_separate(confl_sent_2[:-1], entity_2, pre_attr_state_text)
    
    # Add conflicting sentence prediction and explanation
    if reasoning_depth in ["consistent", "verifiable"]:
        sentence_numbers = (implausible_story['confl_pairs'][0][0] + 1, implausible_story['confl_pairs'][0][1] + 1)
        if conflict_reason_enabled:
            demonstration += conflict_demo_generator_bottom_up(sentence_numbers,
                                                 implausible_story_letter=implausible_story_letter,
                                                 conflict_reason=implausible_story_conflict_reason_generator(implausible_story))
        else:
            demonstration += conflict_demo_generator_bottom_up(sentence_numbers,
                                                 implausible_story_letter=implausible_story_letter,
                                                 conflict_reason=None)
            
    # Add story plausibility prediction
    if story_A["plausible"] == True:
        demonstration += plausibility_demo_generator_bottomup("A")
    elif story_B["plausible"] == True:
        demonstration += plausibility_demo_generator_bottomup("B")

    return demonstration

def get_pre_eff_info_list_for_sent(story, sent_index):
    pre_state_info_list = []
    eff_state_info_list = []
    for entity_idx, entity in enumerate(story['entities']):
        precondition_states = story['states'][entity_idx][sent_index][0]
        effect_states = story['states'][entity_idx][sent_index][1]
        for state_i in range(0, 20):
            attribute = attr_all[state_i]
            pre_state = precondition_states[state_i]
            eff_state = effect_states[state_i]
            if (pre_state == 1 and eff_state == 2) or (pre_state == 2 and eff_state == 1):
                pre_state_info_list.append({"entity": entity, "attribute": attribute, "state": pre_state})
                eff_state_info_list.append({"entity": entity, "attribute": attribute, "state": eff_state})
    if len(pre_state_info_list) == 0 and len(eff_state_info_list) == 0:
        for entity_idx, entity in enumerate(story['entities']):
            precondition_states = story['states'][entity_idx][sent_index][0]
            effect_states = story['states'][entity_idx][sent_index][1]
            for state_i in range(0, 20):
                attribute = attr_all[state_i]
                pre_state = precondition_states[state_i]
                eff_state = effect_states[state_i]
                if pre_state != 0 and pre_state != ATTR_DEFAULT_VALUES[state_i]:
                    pre_state_info_list.append({"entity": entity, "attribute": attribute, "state": pre_state})
                if eff_state != 0 and eff_state != ATTR_DEFAULT_VALUES[state_i]:
                    eff_state_info_list.append({"entity": entity, "attribute": attribute, "state": eff_state})
    if len(pre_state_info_list) == 0:
        for entity_idx, entity in enumerate(story['entities']):
            precondition_states = story['states'][entity_idx][sent_index][0]
            effect_states = story['states'][entity_idx][sent_index][1]
            for state_i in range(0, 20):
                attribute = attr_all[state_i]
                pre_state = precondition_states[state_i]
                eff_state = effect_states[state_i]
                if pre_state != 0:
                    pre_state_info_list.append({"entity": entity, "attribute": attribute, "state": pre_state})
    if len(eff_state_info_list) == 0:
        for entity_idx, entity in enumerate(story['entities']):
            precondition_states = story['states'][entity_idx][sent_index][0]
            effect_states = story['states'][entity_idx][sent_index][1]
            for state_i in range(0, 20):
                attribute = attr_all[state_i]
                pre_state = precondition_states[state_i]
                eff_state = effect_states[state_i]
                if eff_state != 0:
                    eff_state_info_list.append({"entity": entity, "attribute": attribute, "state": eff_state})
    # if len(pre_state_info_list) == 0: print("len(pre_state_info_list) == 0")
    # if len(eff_state_info_list) == 0: print("len(eff_state_info_list) == 0")
    return pre_state_info_list, eff_state_info_list

def story_pair_demo_generator_bottomup_full(story_pair, conflict_reason_enabled=False, reasoning_depth='verifiable'):
    """Generates demo for TRIP two-story task in bottom-up mode."""
    story_A = story_pair[0]
    story_B = story_pair[1]
    demonstration = story_pair_prompt_generator(story_A, story_B)
    # demonstration = ""

    if story_A["plausible"] == True:
        plausible_story = story_A
        implausible_story = story_B
        implausible_story_letter = "B"
    elif story_B["plausible"] == True:
        plausible_story = story_B
        implausible_story = story_A
        implausible_story_letter = "A"

    if reasoning_depth == "verifiable":
        confl_sent_index_1 = implausible_story["confl_pairs"][0][0]
        confl_sent_index_2 = implausible_story["confl_pairs"][0][1]
        confl_sent_1 = implausible_story['sentences'][confl_sent_index_1]
        confl_sent_2 = implausible_story['sentences'][confl_sent_index_2]
        confl_sent1_eff_state_text = ""
        confl_sent2_pre_state_text = ""
        entity_specific_candidates = entity_specific_confl_candidates_finder(implausible_story)
        max_cosine_similarity = 0
        chosen_candidate = None
        if len(entity_specific_candidates) == 0:
            entity_agnostic_candidates = entity_agnostic_confl_candidates_finder(implausible_story)
            for attribute in entity_agnostic_candidates.keys():
                for pair in entity_agnostic_candidates[attribute]:
                    tokens_1 = pair[0]["entity"].split()
                    tokens_2 = pair[1]["entity"].split()
                    sum_tokens_1_embed = np.zeros(50)
                    sum_tokens_2_embed = np.zeros(50)
                    for token in tokens_1:
                        sum_tokens_1_embed += np.array(glove[token])
                    for token in tokens_2:
                        sum_tokens_2_embed += np.array(glove[token])
                    embedding_1 = sum_tokens_1_embed/len(tokens_1) #avg
                    embedding_2 = sum_tokens_2_embed/len(tokens_2) #avg
                    # print(embedding_1, embedding_2)
                    cos_sim = np.dot(embedding_1,embedding_2)/(norm(embedding_1)*norm(embedding_2))
                    if cos_sim > max_cosine_similarity:
                        max_cosine_similarity = cos_sim
                        chosen_candidate = {
                            "entity_1": pair[0]["entity"], 
                            "entity_2": pair[1]["entity"], 
                            "attribute": attribute, 
                            "eff_state": pair[0]["eff_state"], 
                            "pre_state": pair[1]["pre_state"]
                        }
            if chosen_candidate:
                entity_1 = chosen_candidate["entity_1"]
                entity_2 = chosen_candidate["entity_2"]
            else:
                pre_state_info_list = []
                eff_state_info_list = []
                for entity_idx, entity in enumerate(implausible_story['entities']):
                    effect_states = implausible_story['states'][entity_idx][confl_sent_index_1][1]
                    precondition_states = implausible_story['states'][entity_idx][confl_sent_index_2][0]
                    for state_i in range(0, 20):
                        attribute = attr_all[state_i]
                        pre_state = precondition_states[state_i]
                        eff_state = effect_states[state_i]
                        if eff_state != 0 and eff_state != ATTR_DEFAULT_VALUES[state_i]:
                            eff_state_info_list.append({"entity": entity, "attribute": attribute, "eff_state": eff_state})
                        if pre_state != 0 and pre_state != ATTR_DEFAULT_VALUES[state_i]:
                            pre_state_info_list.append({"entity": entity, "attribute": attribute, "pre_state": pre_state})
                random.seed(100)
                random.shuffle(pre_state_info_list)
                random.seed(200)
                random.shuffle(eff_state_info_list)
                pre_state_info = pre_state_info_list[0]
                eff_state_info = eff_state_info_list[0]
                chosen_candidate = {
                    "entity_1": eff_state_info["entity"], 
                    "entity_2": pre_state_info["entity"], 
                    "attribute_1": eff_state_info["attribute"], 
                    "attribute_2": pre_state_info["attribute"],
                    "eff_state": eff_state_info["eff_state"], 
                    "pre_state": pre_state_info["pre_state"]
                }
                entity_1 = chosen_candidate["entity_1"]
                entity_2 = chosen_candidate["entity_2"]
        else:
            chosen_candidate = entity_specific_candidates[0]
            entity_1 = chosen_candidate["entity"]
            entity_2 = chosen_candidate["entity"]
        if "attribute_1" in chosen_candidate.keys() and "attribute_2" in chosen_candidate.keys():
            attribute_1 = chosen_candidate["attribute_1"]
            attribute_2 = chosen_candidate["attribute_2"]
        else:
            attribute_1 = chosen_candidate["attribute"]
            attribute_2 = chosen_candidate["attribute"]
        eff_state = chosen_candidate["eff_state"]
        pre_state = chosen_candidate["pre_state"]
        if attribute_1 == "h_location":
            eff_attr_state_text = h_location_map[eff_state]
        elif attribute_1 == "location":
            eff_attr_state_text = o_location_map[eff_state]
        else:
            eff_attr_state_text = other_attr_map[attribute_1][eff_state - 1]
        if attribute_2 == "h_location":
            pre_attr_state_text = h_location_map[pre_state]
        elif attribute_2 == "location":
            pre_attr_state_text = o_location_map[pre_state]
        else:
            pre_attr_state_text = other_attr_map[attribute_2][pre_state - 1]
        aep_prompt = aep_prompt_generator(confl_sent_1[:-1], entity_1)
        app_prompt = app_prompt_generator(confl_sent_2[:-1], entity_2)
        confl_sent1_eff_state_text = aep_prompt + eff_attr_state_text + ".\n"
        confl_sent2_pre_state_text = app_prompt + pre_attr_state_text + ".\n"
        if story_A["plausible"] == True:
            demo_states_plausible_story = "In story A:\n"
            demo_states_implausible_story = "In story B:\n"
        else:
            demo_states_plausible_story = "In story B:\n"
            demo_states_implausible_story = "In story A:\n"
        for sent_index in range(0, len(plausible_story['sentences'])):
            sentence = plausible_story['sentences'][sent_index]
            pre_state_info_list, eff_state_info_list = get_pre_eff_info_list_for_sent(plausible_story, sent_index)
            random.seed(100)
            random.shuffle(pre_state_info_list)
            random.seed(200)
            random.shuffle(eff_state_info_list)
            pre_state_info = pre_state_info_list[0]
            eff_state_info = eff_state_info_list[0]
            if pre_state_info["attribute"] == "h_location":
                pre_state_text = h_location_map[pre_state_info["state"]]
            elif pre_state_info["attribute"] == "location":
                pre_state_text = o_location_map[pre_state_info["state"]]
            else:
                pre_state_text = other_attr_map[pre_state_info["attribute"]][pre_state_info["state"] - 1]
            if eff_state_info["attribute"] == "h_location":
                eff_state_text = h_location_map[eff_state_info["state"]]
            elif eff_state_info["attribute"] == "location":
                eff_state_text = o_location_map[eff_state_info["state"]]
            else:
                eff_state_text = other_attr_map[eff_state_info["attribute"]][eff_state_info["state"] - 1]
            demo_states_plausible_story = demo_states_plausible_story + "For sentence " + str(sent_index + 1) + ':\n'
            demo_states_plausible_story = demo_states_plausible_story + app_prompt_generator(sentence[:-1], pre_state_info["entity"]) + pre_state_text + ".\n"
            demo_states_plausible_story = demo_states_plausible_story + aep_prompt_generator(sentence[:-1], eff_state_info["entity"]) + eff_state_text + ".\n"
        for sent_index in range(0, len(implausible_story['sentences'])):
            sentence = implausible_story['sentences'][sent_index]
            pre_state_info_list, eff_state_info_list = get_pre_eff_info_list_for_sent(implausible_story, sent_index)
            random.seed(100)
            random.shuffle(pre_state_info_list)
            random.seed(200)
            random.shuffle(eff_state_info_list)
            pre_state_info = pre_state_info_list[0]
            eff_state_info = eff_state_info_list[0]
            if pre_state_info["attribute"] == "h_location":
                pre_state_text = h_location_map[pre_state_info["state"]]
            elif pre_state_info["attribute"] == "location":
                pre_state_text = o_location_map[pre_state_info["state"]]
            else:
                pre_state_text = other_attr_map[pre_state_info["attribute"]][pre_state_info["state"] - 1]
            if eff_state_info["attribute"] == "h_location":
                eff_state_text = h_location_map[eff_state_info["state"]]
            elif eff_state_info["attribute"] == "location":
                eff_state_text = o_location_map[eff_state_info["state"]]
            else:
                eff_state_text = other_attr_map[eff_state_info["attribute"]][eff_state_info["state"] - 1]
            demo_states_implausible_story = demo_states_implausible_story + "For sentence " + str(sent_index + 1) + ':\n'
            if sent_index == confl_sent_index_2:
                demo_states_implausible_story += confl_sent2_pre_state_text
            else:
                demo_states_implausible_story = demo_states_implausible_story + app_prompt_generator(sentence[:-1], pre_state_info["entity"]) + pre_state_text + ".\n"
            if sent_index == confl_sent_index_1:
                demo_states_implausible_story += confl_sent1_eff_state_text
            else:
                demo_states_implausible_story = demo_states_implausible_story + aep_prompt_generator(sentence[:-1], eff_state_info["entity"]) + eff_state_text + ".\n"
        if story_A["plausible"] == True:
            demonstration = demonstration + demo_states_plausible_story
            demonstration = demonstration + demo_states_implausible_story
        else:
            demonstration = demonstration + demo_states_implausible_story
            demonstration = demonstration + demo_states_plausible_story
    
    # Add conflicting sentence prediction and explanation
    if reasoning_depth in ["consistent", "verifiable"]:
        sentence_numbers = (implausible_story['confl_pairs'][0][0] + 1, implausible_story['confl_pairs'][0][1] + 1)
        if conflict_reason_enabled:
            demonstration += conflict_demo_generator_bottom_up(sentence_numbers,
                                                    implausible_story_letter=implausible_story_letter,
                                                    conflict_reason=implausible_story_conflict_reason_generator(implausible_story))
        else:
            demonstration += conflict_demo_generator_bottom_up(sentence_numbers,
                                                    implausible_story_letter=implausible_story_letter,
                                                    conflict_reason=None)

    # Add story plausibility prediction
    if story_A["plausible"] == True:
        demonstration += plausibility_demo_generator_bottomup("A")
    elif story_B["plausible"] == True:
        demonstration += plausibility_demo_generator_bottomup("B")

    return demonstration

def generate_aep_demos(train_dataset, reduce_options):
    if reduce_options == False:
        aep_demo = generate_text_options() # action-effect prediction
    else:
        aep_demo = generate_text_options_reduced_aep()
    options_checklist = []
    for i in range(0, len(train_dataset)):
        stories_info = train_dataset[i]['stories']
        implausible_story = None
        if stories_info[0]['plausible'] == True:
            implausible_story = stories_info[1]
        else:
            implausible_story = stories_info[0]
        confl_sentence_1_idx = implausible_story['confl_pairs'][0][0]
        confl_sentence_2_idx = implausible_story['confl_pairs'][0][1]
        confl_sentence_1 = implausible_story['sentences'][confl_sentence_1_idx]
        candidates = []
        for entity_idx, entity in enumerate(implausible_story['entities']):
            if reduce_options == False:
                if set(options_checklist) == set(all_options):
                    return aep_demo
            else:
                if set(options_checklist) == set(all_options_reduced_aep):
                    return aep_demo
            effect_states = implausible_story['states'][entity_idx][confl_sentence_1_idx][1]
            precondition_states = implausible_story['states'][entity_idx][confl_sentence_2_idx][0]
            for state_i in range(0, 20):
                attribute = attr_all[state_i]
                eff_state = effect_states[state_i]
                pre_state = precondition_states[state_i]
                if (eff_state == 1 and pre_state == 2) or (eff_state == 2 and pre_state == 1):
                    candidates.append({"entity": entity, "attribute": attribute, "eff_state": eff_state, "pre_state": pre_state})
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
            if eff_attr_state_text not in options_checklist:
                if (not reduce_options) or eff_attr_state_text in all_options_reduced_aep:
                    options_checklist.append(eff_attr_state_text)
                    aep_demo += aep_demo_generator(confl_sentence_1[:-1], entity, eff_attr_state_text)
                    break
    for i in range(0, len(train_dataset)):
        stories_info = train_dataset[i]['stories']
        implausible_story = None
        if stories_info[0]['plausible'] == True:
            implausible_story = stories_info[1]
        else:
            implausible_story = stories_info[0]
        confl_sentence_1_idx = implausible_story['confl_pairs'][0][0]
        confl_sentence_1 = implausible_story['sentences'][confl_sentence_1_idx]
        for entity_idx, entity in enumerate(implausible_story['entities']):
            if reduce_options == False:
                if set(options_checklist) == set(all_options):
                    return aep_demo
            else:
                if set(options_checklist) == set(all_options_reduced_aep):
                    return aep_demo
            effect_states = implausible_story['states'][entity_idx][confl_sentence_1_idx][1]
            for state_i in range(0, 20):
                attribute = attr_all[state_i]
                state = effect_states[state_i]
                if state != 0:
                    attr_state_text = None
                    if attribute == "h_location":
                        attr_state_text = h_location_map[state]
                    elif attribute == "location":
                        attr_state_text = o_location_map[state]
                    else:
                        attr_state_text = other_attr_map[attribute][state - 1]
                    if attr_state_text not in options_checklist:
                        if (not reduce_options) or attr_state_text in all_options_reduced_aep:
                            options_checklist.append(attr_state_text)
                            aep_demo += aep_demo_generator(confl_sentence_1[:-1], entity, attr_state_text)
                            break
    return aep_demo

def generate_aep_demos_fully_separate_familiarization(train_dataset, reduce_options):
    if reduce_options == False:
        aep_demo = generate_text_options() # action-effect prediction
    else:
        aep_demo = generate_text_options_reduced_aep()
    options_checklist = []
    for i in range(0, len(train_dataset)):
        stories_info = train_dataset[i]['stories']
        implausible_story = None
        if stories_info[0]['plausible'] == True:
            implausible_story = stories_info[1]
        else:
            implausible_story = stories_info[0]
        confl_sentence_1_idx = implausible_story['confl_pairs'][0][0]
        confl_sentence_2_idx = implausible_story['confl_pairs'][0][1]
        confl_sentence_1 = implausible_story['sentences'][confl_sentence_1_idx]
        candidates = []
        for entity_idx, entity in enumerate(implausible_story['entities']):
            if reduce_options == False:
                if set(options_checklist) == set(all_options):
                    return aep_demo
            else:
                if set(options_checklist) == set(all_options_reduced_aep):
                    return aep_demo
            effect_states = implausible_story['states'][entity_idx][confl_sentence_1_idx][1]
            precondition_states = implausible_story['states'][entity_idx][confl_sentence_2_idx][0]
            for state_i in range(0, 20):
                attribute = attr_all[state_i]
                eff_state = effect_states[state_i]
                pre_state = precondition_states[state_i]
                if (eff_state == 1 and pre_state == 2) or (eff_state == 2 and pre_state == 1):
                    candidates.append({"entity": entity, "attribute": attribute, "eff_state": eff_state, "pre_state": pre_state})
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
            if eff_attr_state_text not in options_checklist:
                if (not reduce_options) or eff_attr_state_text in all_options_reduced_aep:
                    options_checklist.append(eff_attr_state_text)
                    aep_demo += aep_demo_generator_fully_separate_familiarization(confl_sentence_1[:-1], entity, eff_attr_state_text)
                    break
    for i in range(0, len(train_dataset)):
        stories_info = train_dataset[i]['stories']
        implausible_story = None
        if stories_info[0]['plausible'] == True:
            implausible_story = stories_info[1]
        else:
            implausible_story = stories_info[0]
        confl_sentence_1_idx = implausible_story['confl_pairs'][0][0]
        confl_sentence_1 = implausible_story['sentences'][confl_sentence_1_idx]
        for entity_idx, entity in enumerate(implausible_story['entities']):
            if reduce_options == False:
                if set(options_checklist) == set(all_options):
                    return aep_demo
            else:
                if set(options_checklist) == set(all_options_reduced_aep):
                    return aep_demo
            effect_states = implausible_story['states'][entity_idx][confl_sentence_1_idx][1]
            for state_i in range(0, 20):
                attribute = attr_all[state_i]
                state = effect_states[state_i]
                if state != 0:
                    attr_state_text = None
                    if attribute == "h_location":
                        attr_state_text = h_location_map[state]
                    elif attribute == "location":
                        attr_state_text = o_location_map[state]
                    else:
                        attr_state_text = other_attr_map[attribute][state - 1]
                    if attr_state_text not in options_checklist:
                        if (not reduce_options) or attr_state_text in all_options_reduced_aep:
                            options_checklist.append(attr_state_text)
                            aep_demo += aep_demo_generator_fully_separate_familiarization(confl_sentence_1[:-1], entity, attr_state_text)
                            break
    return aep_demo

def generate_app_demos(train_dataset, reduce_options):
    if reduce_options == False:
        app_demo = generate_text_options() # action-effect prediction
    else:
        app_demo = generate_text_options_reduced_app()
    options_checklist = []
    for i in range(0, len(train_dataset)):
        stories_info = train_dataset[i]['stories']
        implausible_story = None
        if stories_info[0]['plausible'] == True:
            implausible_story = stories_info[1]
        else:
            implausible_story = stories_info[0]
        confl_sentence_1_idx = implausible_story['confl_pairs'][0][0]
        confl_sentence_2_idx = implausible_story['confl_pairs'][0][1]
        confl_sentence_2 = implausible_story['sentences'][confl_sentence_2_idx]
        candidates = []
        for entity_idx, entity in enumerate(implausible_story['entities']):
            if reduce_options == False:
                if set(options_checklist) == set(all_options):
                    return app_demo
            else:
                if set(options_checklist) == set(all_options_reduced_app):
                    return app_demo
            effect_states = implausible_story['states'][entity_idx][confl_sentence_1_idx][1]
            precondition_states = implausible_story['states'][entity_idx][confl_sentence_2_idx][0]
            for state_i in range(0, 20):
                attribute = attr_all[state_i]
                eff_state = effect_states[state_i]
                pre_state = precondition_states[state_i]
                if (eff_state == 1 and pre_state == 2) or (eff_state == 2 and pre_state == 1):
                    candidates.append({"entity": entity, "attribute": attribute, "eff_state": eff_state, "pre_state": pre_state})
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
            if pre_attr_state_text not in options_checklist:
                if (not reduce_options) or pre_attr_state_text in all_options_reduced_app:
                    options_checklist.append(pre_attr_state_text)
                    app_demo += app_demo_generator(confl_sentence_2[:-1], entity, pre_attr_state_text)                
                    break
    for i in range(0, len(train_dataset)):
        stories_info = train_dataset[i]['stories']
        implausible_story = None
        if stories_info[0]['plausible'] == True:
            implausible_story = stories_info[1]
        else:
            implausible_story = stories_info[0]
        confl_sentence_2_idx = implausible_story['confl_pairs'][0][1]
        confl_sentence_2 = implausible_story['sentences'][confl_sentence_2_idx]
        for entity_idx, entity in enumerate(implausible_story['entities']):
            if reduce_options == False:
                if set(options_checklist) == set(all_options):
                    return app_demo
            else:
                if set(options_checklist) == set(all_options_reduced_app):
                    return app_demo
            precondition_states = implausible_story['states'][entity_idx][confl_sentence_2_idx][0]
            for state_i in range(0, 20):
                attribute = attr_all[state_i]
                state = precondition_states[state_i]
                if state != 0:
                    attr_state_text = None
                    if attribute == "h_location":
                        attr_state_text = h_location_map[state]
                    elif attribute == "location":
                        attr_state_text = o_location_map[state]
                    else:
                        attr_state_text = other_attr_map[attribute][state - 1]
                    if attr_state_text not in options_checklist:
                        if (not reduce_options) or attr_state_text in all_options_reduced_app:
                            options_checklist.append(attr_state_text)
                            app_demo += app_demo_generator(confl_sentence_2[:-1], entity, attr_state_text)
                            break
    return app_demo

def generate_app_demos_fully_separate_familiarization(train_dataset, reduce_options):
    if reduce_options == False:
        app_demo = generate_text_options() # action-effect prediction
    else:
        app_demo = generate_text_options_reduced_app()
    options_checklist = []
    for i in range(0, len(train_dataset)):
        stories_info = train_dataset[i]['stories']
        implausible_story = None
        if stories_info[0]['plausible'] == True:
            implausible_story = stories_info[1]
        else:
            implausible_story = stories_info[0]
        confl_sentence_1_idx = implausible_story['confl_pairs'][0][0]
        confl_sentence_2_idx = implausible_story['confl_pairs'][0][1]
        confl_sentence_2 = implausible_story['sentences'][confl_sentence_2_idx]
        candidates = []
        for entity_idx, entity in enumerate(implausible_story['entities']):
            if reduce_options == False:
                if set(options_checklist) == set(all_options):
                    return app_demo
            else:
                if set(options_checklist) == set(all_options_reduced_app):
                    return app_demo
            effect_states = implausible_story['states'][entity_idx][confl_sentence_1_idx][1]
            precondition_states = implausible_story['states'][entity_idx][confl_sentence_2_idx][0]
            for state_i in range(0, 20):
                attribute = attr_all[state_i]
                eff_state = effect_states[state_i]
                pre_state = precondition_states[state_i]
                if (eff_state == 1 and pre_state == 2) or (eff_state == 2 and pre_state == 1):
                    candidates.append({"entity": entity, "attribute": attribute, "eff_state": eff_state, "pre_state": pre_state})
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
            if pre_attr_state_text not in options_checklist:
                if (not reduce_options) or pre_attr_state_text in all_options_reduced_app:
                    options_checklist.append(pre_attr_state_text)
                    app_demo += app_demo_generator_fully_separate_familiarization(confl_sentence_2[:-1], entity, pre_attr_state_text)                
                    break
    for i in range(0, len(train_dataset)):
        stories_info = train_dataset[i]['stories']
        implausible_story = None
        if stories_info[0]['plausible'] == True:
            implausible_story = stories_info[1]
        else:
            implausible_story = stories_info[0]
        confl_sentence_2_idx = implausible_story['confl_pairs'][0][1]
        confl_sentence_2 = implausible_story['sentences'][confl_sentence_2_idx]
        for entity_idx, entity in enumerate(implausible_story['entities']):
            if reduce_options == False:
                if set(options_checklist) == set(all_options):
                    return app_demo
            else:
                if set(options_checklist) == set(all_options_reduced_app):
                    return app_demo
            precondition_states = implausible_story['states'][entity_idx][confl_sentence_2_idx][0]
            for state_i in range(0, 20):
                attribute = attr_all[state_i]
                state = precondition_states[state_i]
                if state != 0:
                    attr_state_text = None
                    if attribute == "h_location":
                        attr_state_text = h_location_map[state]
                    elif attribute == "location":
                        attr_state_text = o_location_map[state]
                    else:
                        attr_state_text = other_attr_map[attribute][state - 1]
                    if attr_state_text not in options_checklist:
                        if (not reduce_options) or attr_state_text in all_options_reduced_app:
                            options_checklist.append(attr_state_text)
                            app_demo += app_demo_generator_fully_separate_familiarization(confl_sentence_2[:-1], entity, attr_state_text)
                            break
    return app_demo

def plausible_story_extractor_bottomup(texts_sep_by_newline):
    """Identifies plausibility label from LM generated text in bottom-up mode."""
    plausible_story = 'A'
    # print(texts_sep_by_newline)
    for text in texts_sep_by_newline:
        confl_sents_finder = re.findall(r"Therefore, story (.*) is more plausible.", text)
        if len(confl_sents_finder) >= 1:
            plausible_story = confl_sents_finder[0]
    return plausible_story

def confl_pairs_extractor_topdown(raw_generated_tokens):
    """Identifies conflicting sentence indices from LM generated text in top-down mode."""
    raw_generated_text = " ".join(raw_generated_tokens)
    conflict_finder = re.findall(r"(.*) and (.*) conflict with each other", raw_generated_text)
    confl_pairs = []
    if len(conflict_finder) >= 1:
        confl_pair = []
        conflict_strings = conflict_finder[0]
        for conflict_string in conflict_strings:
            for token in conflict_string.split():
                if token.isdigit():
                    confl_pair.append(int(token) - 1)
        confl_pairs.append(confl_pair[:2])
    return confl_pairs

def plausibility_extractor_cot(raw_generated_tokens):
    plaus_finder = re.findall(r"Therefore, Story (.*) is more plausible", raw_generated_tokens)
    # print(plaus_finder)
    if len(plaus_finder) == 0: return 'A'
    return plaus_finder[0]

def confl_pairs_extractor_bottomup(texts_sep_by_newline):
    """Identifies conflicting sentence indices from LM generated text in bottom-up mode."""
    confl_pairs = []
    # print(texts_sep_by_newline)
    for text in texts_sep_by_newline:
        confl_sents_finder = re.findall(r"Therefore, sentences (.*) and (.*) conflict with each other", text)
        if len(confl_sents_finder) >= 1:
            if confl_sents_finder[0][0].isdigit() and confl_sents_finder[0][1].isdigit():
                number_1 = int(confl_sents_finder[0][0]) - 1
                number_2 = int(confl_sents_finder[0][1]) - 1
                if number_1 > number_2: number_1, number_2 = number_2, number_1
                confl_pairs.append([number_1, number_2])
                break
    return confl_pairs

def physical_states_extractor_topdown(raw_generated_text):
    """Uses regular expressions to extract physical states from LM generated text in top-down mode."""
    sent_1_entity = None
    sent_1_eff = None
    sent_2_entity = None
    sent_2_pre = None
    # NOTE: can remove "what is the state of"/"what was the state of" from below when testing with old combined results
    # sent_entity_eff_finder = re.findall(r"what is the state of (.*)\? (.*) is now (.*)", raw_generated_text) 
    # sent_entity_pre_finder = re.findall(r"what was the state of (.*)\? (.*) was (.*)", raw_generated_text)
    sent_entity_eff_finder = re.findall(r"(.*)\? (.*) (?:is|are) now (.*)", raw_generated_text) 
    sent_entity_pre_finder = re.findall(r"(.*)\? (.*) (?:was|were) (.*)", raw_generated_text)

    if len(sent_entity_eff_finder) >= 1:
        entity_state_1_tokens = sent_entity_eff_finder[0][1].split()
        if entity_state_1_tokens[0] == 'The':
            sent_1_entity = ' '.join(entity_state_1_tokens[1:]).replace('?','')
        else:
            sent_1_entity = ' '.join(entity_state_1_tokens[0:]).replace('?','')
        sent_1_eff = sent_entity_eff_finder[0][2].replace('.','')
    else:
        sent_1_entity = "None"
        sent_1_eff = "None"

    if len(sent_entity_pre_finder) >= 1:
        entity_state_2_tokens = sent_entity_pre_finder[0][1].split()
        if entity_state_2_tokens[0] == 'The':
            sent_2_entity = ' '.join(entity_state_2_tokens[1:]).replace('?','')
        else:
            sent_2_entity = ' '.join(entity_state_2_tokens[0:]).replace('?','')
        sent_2_pre = sent_entity_pre_finder[0][2].replace('.','')
    else:
        sent_2_entity = "None"
        sent_2_pre = "None"
    return sent_1_entity, sent_1_eff, sent_2_entity, sent_2_pre

def physical_states_extractor_cot(raw_generated_text):
    """Uses regular expressions to extract physical states from LM generated text in top-down mode."""
    sent_1_entity = None
    sent_1_eff = None
    sent_2_entity = None
    sent_2_pre = None
    # NOTE: can remove "what is the state of"/"what was the state of" from below when testing with old combined results
    # sent_entity_eff_finder = re.findall(r"what is the state of (.*)\? (.*) is now (.*)", raw_generated_text) 
    # sent_entity_pre_finder = re.findall(r"what was the state of (.*)\? (.*) was (.*)", raw_generated_text)
    sent_entity_eff_finder = re.findall(r"After, what is the state of the (.*)\? (.*) (?:is|are) now (.*).", raw_generated_text) 
    sent_entity_pre_finder = re.findall(r"Before, what was the state of the (.*)\? (.*) (?:was|were) (.*).", raw_generated_text)

    if len(sent_entity_eff_finder) >= 1:
        entity_state_1_tokens = sent_entity_eff_finder[0][1].split()
        if entity_state_1_tokens[0] == 'The':
            sent_1_entity = ' '.join(entity_state_1_tokens[1:]).replace('?','')
        else:
            sent_1_entity = ' '.join(entity_state_1_tokens[0:]).replace('?','')
        sent_1_eff = sent_entity_eff_finder[0][2].replace('.','')
    else:
        sent_1_entity = "None"
        sent_1_eff = "None"

    if len(sent_entity_pre_finder) >= 1:
        entity_state_2_tokens = sent_entity_pre_finder[0][1].split()
        if entity_state_2_tokens[0] == 'The':
            sent_2_entity = ' '.join(entity_state_2_tokens[1:]).replace('?','')
        else:
            sent_2_entity = ' '.join(entity_state_2_tokens[0:]).replace('?','')
        sent_2_pre = sent_entity_pre_finder[0][2].replace('.','')
    else:
        sent_2_entity = "None"
        sent_2_pre = "None"
    # print(sent_1_entity, sent_1_eff, sent_2_entity, sent_2_pre)
    return sent_1_entity, sent_1_eff, sent_2_entity, sent_2_pre

def physical_states_extractor_bottomup_full(texts_sep_by_newline, n_sentences):
    """Uses regular expressions to extract physical states from LM generated text in bottom-up mode."""
    sent_1_entity = "None"
    sent_1_eff = "None"
    sent_2_entity = "None"
    sent_2_pre = "None"
    confl_pairs = confl_pairs_extractor_bottomup(texts_sep_by_newline)
    plausible_story = plausible_story_extractor_bottomup(texts_sep_by_newline)
    confl_sent1_idx = confl_pairs[0][0]
    confl_sent2_idx = confl_pairs[0][1]
    confl_eff_line_idx = 0
    confl_pre_line_idx = 0
    if plausible_story == "B":
        confl_eff_line_idx = 1 + 3 * confl_sent1_idx + 2
        confl_pre_line_idx = 1 + 3 * confl_sent2_idx + 1
    else:
        confl_eff_line_idx = 1 + 3 * n_sentences + 1 + 3 * confl_sent1_idx + 2
        confl_pre_line_idx = 1 + 3 * n_sentences + 1 + 3 * confl_sent2_idx + 1
    confl_eff_line = texts_sep_by_newline[confl_eff_line_idx]
    confl_pre_line = texts_sep_by_newline[confl_pre_line_idx]
    # print(confl_eff_line)
    # print(confl_pre_line)
    sent_entity_eff_finder = re.findall(r"After (.*)\? (.*) (?:is|are) now (.*)", confl_eff_line)
    sent_entity_pre_finder = re.findall(r"Before (.*)\? (.*) (?:was|were) (.*)", confl_pre_line)

    # print(sent_entity_eff_finder)
    # print(sent_entity_pre_finder)
    if len(sent_entity_eff_finder) >= 1:
        entity_state_1_tokens = sent_entity_eff_finder[0][1].split()
        if entity_state_1_tokens[0] == 'The':
            sent_1_entity = ' '.join(entity_state_1_tokens[1:]).replace('?','')
        else:
            sent_1_entity = ' '.join(entity_state_1_tokens[0:]).replace('?','')
        sent_1_eff = sent_entity_eff_finder[0][2].replace('.','')
    else:
        sent_1_entity = "None"
        sent_1_eff = "None"
    if len(sent_entity_pre_finder) >= 1:
        entity_state_2_tokens = sent_entity_pre_finder[0][1].split()
        if entity_state_2_tokens[0] == 'The':
            sent_2_entity = ' '.join(entity_state_2_tokens[1:]).replace('?','')
        else:
            sent_2_entity = ' '.join(entity_state_2_tokens[0:]).replace('?','')
        sent_2_pre = sent_entity_pre_finder[0][2].replace('.','')
    else:
        sent_2_entity = "None"
        sent_2_pre = "None"
    # print(sent_1_entity)
    # print(sent_1_eff)
    # print(sent_2_entity)
    # print(sent_2_pre)
    return sent_1_entity, sent_1_eff, sent_2_entity, sent_2_pre

def physical_states_extractor_bottomup_compact(texts_sep_by_newline):
    """Uses regular expressions to extract physical states from LM generated text in bottom-up mode."""
    sent_1_entity = "None"
    sent_1_eff = "None"
    sent_2_entity = "None"
    sent_2_pre = "None"
    
    for text in texts_sep_by_newline: 
        sent_entity_eff_finder = re.findall(r"(.*)\? (.*) (?:is|are) now (.*)", text) 
        sent_entity_pre_finder = re.findall(r"(.*)\? (.*) (?:was|were) (.*)", text)
        if len(sent_entity_eff_finder) >= 1:
            entity_state_1_tokens = sent_entity_eff_finder[0][1].split()
            if entity_state_1_tokens[0] == 'The':
                sent_1_entity = ' '.join(entity_state_1_tokens[1:]).replace('?','')
            else:
                sent_1_entity = ' '.join(entity_state_1_tokens[0:]).replace('?','')
            sent_1_eff = sent_entity_eff_finder[0][2].replace('.','')

        if len(sent_entity_pre_finder) >= 1:
            entity_state_2_tokens = sent_entity_pre_finder[0][1].split()
            if entity_state_2_tokens[0] == 'The':
                sent_2_entity = ' '.join(entity_state_2_tokens[1:]).replace('?','')
            else:
                sent_2_entity = ' '.join(entity_state_2_tokens[0:]).replace('?','')
            sent_2_pre = sent_entity_pre_finder[0][2].replace('.','')
    # print(sent_1_entity, sent_1_eff, sent_2_entity, sent_2_pre)
    return sent_1_entity, sent_1_eff, sent_2_entity, sent_2_pre

def create_state(attr_state_text):
    if attr_state_text == "irrelevant":
        return [0] * 20
    if attr_state_text not in attr_state_map.keys():
        return [0] * 20
    state_idx = attr_state_map[attr_state_text][0]
    state_value = attr_state_map[attr_state_text][1]
    state = [0] * 20
    state[state_idx] = state_value
    return state


def state_predict(prediction, label):
    # Check verifiability of one set of predicted states
    # ATTR_DEFAULT_VALUES = [0, 2, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0]
    correct = True
    found_nondefault = False
    for i in range(0, 20):
        if prediction[i] != ATTR_DEFAULT_VALUES[i] and prediction[i] > 0:
            found_nondefault = True
            if prediction[i] != label[i]:
                correct = False
                break
    return correct, found_nondefault

def extract_and_combine_physical_states_topdown(story_generated_text, prediction_obj, story_entities, sent_1_entities, sent_2_entities):
    sent_1_entity, aep_generated_text, sent_2_entity, app_generated_text = physical_states_extractor_topdown(story_generated_text)

    # Map extracted entities to mentioned entities in target sentences, e.g., in cases where the extracted entity isn't labeled in the sentence.
    # (We do this in the TRIP baselines because we aren't worried about coreference resolution - just assume we can reliably extract the target entities and evaluate based on their labels)
    if not is_human(sent_1_entity):
        sent_1_entity_idx = glove_similarity_match(
            glove, 
            sent_1_entity, 
            [ent if not is_human(ent) else "someone" for ent in sent_1_entities]
        )
        sent_1_entity = sent_1_entities[sent_1_entity_idx]
    if not is_human(sent_2_entity):
        sent_2_entity_idx = glove_similarity_match(
            glove, 
            sent_2_entity, 
            [ent if not is_human(ent) else "someone" for ent in sent_2_entities]
        )
        sent_2_entity = sent_2_entities[sent_2_entity_idx]

    if (aep_generated_text == "disappeared" or aep_generated_text == "moved somewhere new" or aep_generated_text == "dry" or aep_generated_text == "wet" or aep_generated_text == "dirty" or aep_generated_text == "clean") and is_human(sent_1_entity):
        aep_generated_text = aep_generated_text + "_h"
    if (app_generated_text == "disappeared" or app_generated_text == "moved somewhere new" or app_generated_text == "dry" or app_generated_text == "wet" or app_generated_text == "dirty" or app_generated_text == "clean") and is_human(sent_2_entity):
        app_generated_text = app_generated_text + "_h"
    effect_states_sent1 = create_state(aep_generated_text)
    pre_condition_states_sent2 = create_state(app_generated_text)
    return_physical_states = []
    if sent_1_entity in story_entities:
        sent_1_entity_index = story_entities.index(sent_1_entity)
    else:
        sent_1_entity_index = 0
    if sent_2_entity in story_entities:
        sent_2_entity_index = story_entities.index(sent_2_entity)
    else:
        sent_2_entity_index = 0
    return_physical_states.append([sent_1_entity_index, prediction_obj['confl_pairs'][0][0] if len(prediction_obj['confl_pairs']) > 0 else None, effect_states_sent1])
    return_physical_states.append([sent_2_entity_index, prediction_obj['confl_pairs'][0][1] if len(prediction_obj['confl_pairs']) > 0 else None, pre_condition_states_sent2])
    return return_physical_states

def extract_and_combine_physical_states_cot(story_generated_text, prediction_obj, story_entities, sent_1_entities, sent_2_entities):
    sent_1_entity, aep_generated_text, sent_2_entity, app_generated_text = physical_states_extractor_cot(story_generated_text)
    # print(sent_1_entity)
    # print(aep_generated_text)
    # print(sent_2_entity)
    # print(app_generated_text)

    # Map extracted entities to mentioned entities in target sentences, e.g., in cases where the extracted entity isn't labeled in the sentence.
    # (We do this in the TRIP baselines because we aren't worried about coreference resolution - just assume we can reliably extract the target entities and evaluate based on their labels)
    if not is_human(sent_1_entity):
        sent_1_entity_idx = glove_similarity_match(
            glove, 
            sent_1_entity, 
            [ent if not is_human(ent) else "someone" for ent in sent_1_entities]
        )
        sent_1_entity = sent_1_entities[sent_1_entity_idx]
    if not is_human(sent_2_entity):
        sent_2_entity_idx = glove_similarity_match(
            glove, 
            sent_2_entity, 
            [ent if not is_human(ent) else "someone" for ent in sent_2_entities]
        )
        sent_2_entity = sent_2_entities[sent_2_entity_idx]

    if (aep_generated_text == "disappeared" or aep_generated_text == "moved somewhere new" or aep_generated_text == "dry" or aep_generated_text == "wet" or aep_generated_text == "dirty" or aep_generated_text == "clean") and is_human(sent_1_entity):
        aep_generated_text = aep_generated_text + "_h"
    if (app_generated_text == "disappeared" or app_generated_text == "moved somewhere new" or app_generated_text == "dry" or app_generated_text == "wet" or app_generated_text == "dirty" or app_generated_text == "clean") and is_human(sent_2_entity):
        app_generated_text = app_generated_text + "_h"
    effect_states_sent1 = create_state(aep_generated_text)
    pre_condition_states_sent2 = create_state(app_generated_text)
    return_physical_states = []
    if sent_1_entity in story_entities:
        sent_1_entity_index = story_entities.index(sent_1_entity)
    else:
        sent_1_entity_index = 0
    if sent_2_entity in story_entities:
        sent_2_entity_index = story_entities.index(sent_2_entity)
    else:
        sent_2_entity_index = 0
    return_physical_states.append([sent_1_entity_index, prediction_obj['confl_pairs'][0][0] if len(prediction_obj['confl_pairs']) > 0 else None, effect_states_sent1])
    return_physical_states.append([sent_2_entity_index, prediction_obj['confl_pairs'][0][1] if len(prediction_obj['confl_pairs']) > 0 else None, pre_condition_states_sent2])
    return return_physical_states

def add_trip_preds_topdown(prediction_obj, implausible_story, story_generated_text):
    """Adds tiered predictions to prediction_obj extracted from story_generated_text."""
    prediction_raw_tokens = story_generated_text.split(' ')
    if len(prediction_raw_tokens) > 1:
        predicted_plausible_story = prediction_raw_tokens[1] # Just take second token as the plausible story
        prediction_obj['plausible_story'] =  predicted_plausible_story
        predicted_confl_pair = confl_pairs_extractor_topdown(prediction_raw_tokens)
        prediction_obj['confl_pairs'] = predicted_confl_pair
        if len(prediction_obj['confl_pairs']) > 0:
            prediction_obj['physical_states'] = extract_and_combine_physical_states_topdown(
                story_generated_text, 
                prediction_obj, 
                implausible_story['entities'], 
                implausible_story['entities_by_sentence'][prediction_obj['confl_pairs'][0][0]],
                implausible_story['entities_by_sentence'][prediction_obj['confl_pairs'][0][1]],
            )
    else:
        prediction_obj['plausible_story'] = "A"
        prediction_obj['confl_pairs'] = []
        prediction_obj['physical_states'] = []

def add_trip_preds_topdown_ask_implausible(prediction_obj, implausible_story, story_generated_text):
    """Adds tiered predictions to prediction_obj extracted from story_generated_text."""
    prediction_raw_tokens = story_generated_text.split(' ')
    if len(prediction_raw_tokens) > 1:
        predicted_implausible_story = prediction_raw_tokens[1] # Just take second token as the plausible story
        if predicted_implausible_story == 'A':
            prediction_obj['plausible_story'] =  'B'
        else:
            prediction_obj['plausible_story'] =  'A'
        predicted_confl_pair = confl_pairs_extractor_topdown(prediction_raw_tokens)
        prediction_obj['confl_pairs'] = predicted_confl_pair
        if len(prediction_obj['confl_pairs']) > 0:
            prediction_obj['physical_states'] = extract_and_combine_physical_states_topdown(
                story_generated_text, 
                prediction_obj, 
                implausible_story['entities'], 
                implausible_story['entities_by_sentence'][prediction_obj['confl_pairs'][0][0]],
                implausible_story['entities_by_sentence'][prediction_obj['confl_pairs'][0][1]],
            )
    else:
        prediction_obj['plausible_story'] = "A"
        prediction_obj['confl_pairs'] = []
        prediction_obj['physical_states'] = []

def extract_and_combine_physical_states_bottomup_full(texts_sep_by_newline, prediction_obj, implausible_story, sent_1_entities, sent_2_entities):
    story_entities = implausible_story['entities']
    n_sentences = len(implausible_story["sentences"])
    sent_1_entity, aep_generated_text, sent_2_entity, app_generated_text = physical_states_extractor_bottomup_full(texts_sep_by_newline, n_sentences)

    # Map extracted entities to mentioned entities in target sentences, e.g., in cases where the extracted entity isn't labeled in the sentence.
    # (We do this in the TRIP baselines because we aren't worried about coreference resolution - just assume we can reliably extract the target entities and evaluate based on their labels)
    if not is_human(sent_1_entity):
        sent_1_entity_idx = glove_similarity_match(
            glove, 
            sent_1_entity, 
            [ent if not is_human(ent) else "someone" for ent in sent_1_entities]
        )
        sent_1_entity = sent_1_entities[sent_1_entity_idx]
    if not is_human(sent_2_entity):
        sent_2_entity_idx = glove_similarity_match(
            glove, 
            sent_2_entity, 
            [ent if not is_human(ent) else "someone" for ent in sent_2_entities]
        )
        sent_2_entity = sent_2_entities[sent_2_entity_idx]
    
    if (aep_generated_text == "disappeared" or aep_generated_text == "moved somewhere new" or aep_generated_text == "dry" or aep_generated_text == "wet" or aep_generated_text == "dirty" or aep_generated_text == "clean") and is_human(sent_1_entity):
        aep_generated_text = aep_generated_text + "_h"
    if (app_generated_text == "disappeared" or app_generated_text == "moved somewhere new" or app_generated_text == "dry" or app_generated_text == "wet" or app_generated_text == "dirty" or app_generated_text == "clean") and is_human(sent_2_entity):
        app_generated_text = app_generated_text + "_h"
    effect_states_sent1 = create_state(aep_generated_text)
    pre_condition_states_sent2 = create_state(app_generated_text)
    return_physical_states = []
    if sent_1_entity in story_entities:
        sent_1_entity_index = story_entities.index(sent_1_entity)
    else:
        sent_1_entity_index = 0
    if sent_2_entity in story_entities:
        sent_2_entity_index = story_entities.index(sent_2_entity)
    else:
        sent_2_entity_index = 0
    return_physical_states.append([sent_1_entity_index, prediction_obj['confl_pairs'][0][0] if len(prediction_obj['confl_pairs']) > 0 else None, effect_states_sent1])
    return_physical_states.append([sent_2_entity_index, prediction_obj['confl_pairs'][0][1] if len(prediction_obj['confl_pairs']) > 0 else None, pre_condition_states_sent2])
    return return_physical_states

def extract_and_combine_physical_states_bottomup_compact(texts_sep_by_newline, prediction_obj, implausible_story, sent_1_entities, sent_2_entities):
    story_entities = implausible_story['entities']
    n_sentences = len(implausible_story["sentences"])
    sent_1_entity, aep_generated_text, sent_2_entity, app_generated_text = physical_states_extractor_bottomup_compact(texts_sep_by_newline)
    # print(sent_1_entity, aep_generated_text, sent_2_entity, app_generated_text)

    # Map extracted entities to mentioned entities in target sentences, e.g., in cases where the extracted entity isn't labeled in the sentence.
    # (We do this in the TRIP baselines because we aren't worried about coreference resolution - just assume we can reliably extract the target entities and evaluate based on their labels)
    if not is_human(sent_1_entity):
        sent_1_entity_idx = glove_similarity_match(
            glove, 
            sent_1_entity, 
            [ent if not is_human(ent) else "someone" for ent in sent_1_entities]
        )
        sent_1_entity = sent_1_entities[sent_1_entity_idx]
    if not is_human(sent_2_entity):
        sent_2_entity_idx = glove_similarity_match(
            glove, 
            sent_2_entity, 
            [ent if not is_human(ent) else "someone" for ent in sent_2_entities]
        )
        sent_2_entity = sent_2_entities[sent_2_entity_idx]
    
    if (aep_generated_text == "disappeared" or aep_generated_text == "moved somewhere new" or aep_generated_text == "dry" or aep_generated_text == "wet" or aep_generated_text == "dirty" or aep_generated_text == "clean") and is_human(sent_1_entity):
        aep_generated_text = aep_generated_text + "_h"
    if (app_generated_text == "disappeared" or app_generated_text == "moved somewhere new" or app_generated_text == "dry" or app_generated_text == "wet" or app_generated_text == "dirty" or app_generated_text == "clean") and is_human(sent_2_entity):
        app_generated_text = app_generated_text + "_h"
    effect_states_sent1 = create_state(aep_generated_text)
    pre_condition_states_sent2 = create_state(app_generated_text)
    return_physical_states = []
    if sent_1_entity in story_entities:
        sent_1_entity_index = story_entities.index(sent_1_entity)
    else:
        sent_1_entity_index = 0
    if sent_2_entity in story_entities:
        sent_2_entity_index = story_entities.index(sent_2_entity)
    else:
        sent_2_entity_index = 0
    return_physical_states.append([sent_1_entity_index, prediction_obj['confl_pairs'][0][0] if len(prediction_obj['confl_pairs']) > 0 else None, effect_states_sent1])
    return_physical_states.append([sent_2_entity_index, prediction_obj['confl_pairs'][0][1] if len(prediction_obj['confl_pairs']) > 0 else None, pre_condition_states_sent2])
    return return_physical_states

def add_trip_preds_bottomup_full(prediction_obj, implausible_story, story_generated_text):
    """Adds tiered predictions to prediction_obj extracted from story_generated_text."""
    generated_texts_sep_by_newline = story_generated_text.split('\n')
    predicted_plausible_story = plausible_story_extractor_bottomup(generated_texts_sep_by_newline) # Just take second token as the plausible story
    prediction_obj['plausible_story'] =  predicted_plausible_story
    predicted_confl_pair = confl_pairs_extractor_bottomup(generated_texts_sep_by_newline)
    prediction_obj['confl_pairs'] = predicted_confl_pair
    if len(prediction_obj['confl_pairs']) > 0:
        prediction_obj['physical_states'] = extract_and_combine_physical_states_bottomup_full(
            generated_texts_sep_by_newline, 
            prediction_obj, 
            implausible_story, 
            implausible_story['entities_by_sentence'][prediction_obj['confl_pairs'][0][0]],
            implausible_story['entities_by_sentence'][prediction_obj['confl_pairs'][0][1]],
        )

def add_trip_preds_bottomup_compact(prediction_obj, implausible_story, story_generated_text):
    """Adds tiered predictions to prediction_obj extracted from story_generated_text."""
    generated_texts_sep_by_newline = story_generated_text.split('\n')
    predicted_plausible_story = plausible_story_extractor_bottomup(generated_texts_sep_by_newline)
    prediction_obj['plausible_story'] =  predicted_plausible_story
    predicted_confl_pair = confl_pairs_extractor_bottomup(generated_texts_sep_by_newline)
    prediction_obj['confl_pairs'] = predicted_confl_pair
    if len(prediction_obj['confl_pairs']) > 0:
        prediction_obj['physical_states'] = extract_and_combine_physical_states_bottomup_compact(
            generated_texts_sep_by_newline, 
            prediction_obj, 
            implausible_story, 
            implausible_story['entities_by_sentence'][prediction_obj['confl_pairs'][0][0]],
            implausible_story['entities_by_sentence'][prediction_obj['confl_pairs'][0][1]],
        )