import numpy as np
import json
import os

def condense_multiple_conflict_pairs(conflict_sents, breakpoint):
    return [max([s for s in conflict_sents if s < breakpoint]), breakpoint]

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
    
def entity_specific_confl_candidates_finder(implausible_story):
    """Find candidates with a mismatch on effect state and precondition state specific to the entity name"""
    """If the returned list is empty, there's no explicit conflict from physical states (on the same attribute)"""
    candidates = []
    confl_sent_index_1 = implausible_story["confl_pairs"][0][0]
    confl_sent_index_2 = implausible_story["confl_pairs"][0][1]
    # print("len(story):", len(implausible_story["sentences"]))
    # print(confl_sent_index_1, confl_sent_index_2)
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

# TODO: Add changed_state with entity_name
def raw_dict_to_dataset(raw_dict, exclude_multi_confl=False):
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
                have_entity_specific_confl = False
                pair_plausible_story = None
                temp_implausible_story = {}
                pair_confl_pairs = None
                for story_index, story in enumerate(object["stories"]):
                    if len(story['sentences']) != len(story['states']):
                        print('missed a story!')
                        have_all_states = False                  
                    story_json_data = {}
                    story_json_data["sentences"] = story["sentences"]
                    story_json_data["confl_pairs"] = story["confl_pairs"]
                    story_json_data["confl_sents"] = story["confl_sents"]
                    story_json_data["breakpoint"] = story["breakpoint"]
                    if len(story_json_data["confl_pairs"]) > 1:
                        # Use heuristic to condense multiple conflict pairs into 1
                        story_json_data["confl_pairs"] = [condense_multiple_conflict_pairs(story_json_data["confl_sents"], story_json_data["breakpoint"])]
                        multi_conflicts += 1
                        # if story_json_data["confl_pairs"][0][1] >= len(story_json_data["sentences"]):
                        #     print("problem happens")
                        #     print(story_json_data["confl_sents"])
                        #     print(story_json_data["breakpoint"])
                        #     print(story_json_data["confl_pairs"])
                    if len(story_json_data["confl_pairs"]) == 0:
                        story_json_data["plausible"] = 1
                        pair_plausible_story = story_index
                    else:
                        story_json_data["plausible"] = 0
                        pair_confl_pairs = story_json_data["confl_pairs"]
                        temp_implausible_story["sentences"] = story["sentences"]
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
                    if story_json_data["plausible"] == 0: temp_implausible_story["entities"] = entities
                    states = []
                    for _ in range(len(entities)):
                        sentences_state = []
                        for _ in range(len(story_json_data["sentences"])):
                            sentences_state.append([[0] * 20, [0] * 20])
                        states.append(sentences_state)
                    for sentence_i in range(0, len(story["sentences"])):
                        if sentence_i < len(story["states"]): # handle incomplete data
                            sentence_states = story["states"][sentence_i]
                            for attribute in sentence_states:
                                for entity_state in sentence_states[attribute]:
                                    entity, state = entity_state
                                    if attribute == "h_location" or attribute == "location":
                                        states[entities.index(entity)][sentence_i][0][attributes.index(attribute)] = state
                                        states[entities.index(entity)][sentence_i][1][attributes.index(attribute)] = state
                                    else:
                                        states[entities.index(entity)][sentence_i][0][attributes.index(attribute)] = other_attributes_map[state][0]
                                        states[entities.index(entity)][sentence_i][1][attributes.index(attribute)] = other_attributes_map[state][1]
                    story_json_data["states"] = states
                    if story_json_data["plausible"] == 0: temp_implausible_story["states"] = states
                    stories.append(story_json_data)
                    if story_json_data["plausible"] == 0:
                        entity_specific_confl_candidates = entity_specific_confl_candidates_finder(story_json_data)
                        if len(entity_specific_confl_candidates) > 0:
                            have_entity_specific_confl = True
                            entity_specific_confl_candidate = entity_specific_confl_candidates[0]
                            entity_index = story_json_data["entities"].index(entity_specific_confl_candidate["entity"])
                            attribute_index = attr_all.index(entity_specific_confl_candidate["attribute"])
                            effect_state = entity_specific_confl_candidate["eff_state"]
                            precondition_state = entity_specific_confl_candidate["pre_state"]
                if have_entity_specific_confl:
                    pass
                else:
                    stories_json_data['example_id'] = object['example_id']
                    stories_json_data["stories"] = stories
                    stories_json_data["pair_entities"] = temp_implausible_story["entities"]
                    stories_json_data["pair_plausible_story"] = pair_plausible_story
                    stories_json_data["pair_confl_pairs"] = pair_confl_pairs
                    stories_json_data["pair_states"] = temp_implausible_story["states"]
                    if have_all_states:
                        dataset.append(stories_json_data)
        else:
            raise ValueError("an example is missing labeled physical states")
    print("%s instances with multiple conflicts" % multi_conflicts)
    return dataset

def make_trip_dataset():
    raw_all_data = json.load(open("trip.json", "r"))
    raw_test = raw_all_data[0]["test"] # cloze
    trip_dataset = {}
    trip_dataset["test_implicit"] = raw_dict_to_dataset(raw_test, exclude_multi_confl=False)

    print("# pairs of stories in test_implicit_trip data:", len(trip_dataset["test_implicit"]))

    return trip_dataset

def is_human(entity):
    return (entity[0].isupper() and entity != 'TV')

def write_to_files(trip_dataset):
    if os.path.isfile("test_implicit_trip.json"):
        print("(test_implicit_trip.json exists) Write process suspended. Please delete test_implicit_trip.json and try again.")
    else:
        for json_obj in trip_dataset["test_implicit"]:
            with open("test_implicit_trip.json", "a") as output_file:
                output_file.write(json.dumps(json_obj) + "\n")

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

if __name__ == "__main__":
    trip_dataset = make_trip_dataset()
    write_to_files(trip_dataset)
