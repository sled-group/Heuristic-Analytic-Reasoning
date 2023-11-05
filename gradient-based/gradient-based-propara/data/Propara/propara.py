import json
import os
from tabulate import tabulate
from sklearn.model_selection import train_test_split
import random

def get_conversions(story):
    sentences = story["sentence_texts"]
    participants = story["participants"]
    states_for_participants = story["states"]
    conversions = []
    for participant_i in range(0, len(participants)):
        for state_step in range(0, len(sentences)): # to avoid state_step + 1 out of index
            proposal_before_conversion_state = states_for_participants[participant_i][state_step]
            if proposal_before_conversion_state not in ['-', '?']:
                proposal_after_conversion_state = states_for_participants[participant_i][state_step + 1]
                if proposal_after_conversion_state == '-':
                    for participant_j in range(0, len(participants)):
                        if participant_j != participant_i:
                            if states_for_participants[participant_j][state_step + 1] not in ['-', '?']:
                                if states_for_participants[participant_j][state_step] == '-':
                                    conversions.append({"participant_converted": participants[participant_i], "participant_converted_to": participants[participant_j], "state_converted_from": state_step, "state_converted_to": state_step + 1})
    return conversions

def make_table(story):
    data = []
    sentences = story["sentence_texts"]
    participants = story["participants"]
    states_for_participants = story["states"]
    for state_step in range(0, len(sentences)):
        if state_step == len(sentences) - 1:
            data.append(["", "state" + str(state_step)])
            for participant_i in range(0, len(participants)):
                data[-1].append(states_for_participants[participant_i][state_step])
            
            data.append([sentences[state_step], ""])
            for participant_i in range(0, len(participants)):
                data[-1].append("")
            
            data.append(["", "state" + str(state_step + 1)])
            for participant_i in range(0, len(participants)):
                data[-1].append(states_for_participants[participant_i][state_step + 1])
        else:
            data.append(["", "state" + str(state_step)])
            for participant_i in range(0, len(participants)):
                data[-1].append(states_for_participants[participant_i][state_step])
            
            data.append([sentences[state_step], ""])
            for participant_i in range(0, len(participants)):
                data[-1].append("")
    headers = ["sentence", ""]    
    for participant in participants:
        headers.append(participant)
    table = tabulate(data, headers, tablefmt="pretty")
    return table

def inspect_dataset_conversions(raw_dict):
    for story_i in range(0, len(raw_dict)):
        conversions = get_conversions(raw_dict[story_i])
        if len(conversions) > 0:
            print("Story Index", story_i)
            print(make_table(raw_dict[story_i]))
            for conversion in conversions:
                print(conversion)
            print('-' * 100)

def merge_conversions(conversions):
    merged_conversions = {}
    for conversion in conversions:
        if conversion["participant_converted"] in merged_conversions.keys():
            merged_conversions[conversion["participant_converted"]].append({"participant_converted_to": conversion["participant_converted_to"], "state_converted_from": conversion["state_converted_from"], "state_converted_to": conversion["state_converted_to"]})
        else:
            merged_conversions[conversion["participant_converted"]] = [{"participant_converted_to": conversion["participant_converted_to"], "state_converted_from": conversion["state_converted_from"], "state_converted_to": conversion["state_converted_to"]}]
    return merged_conversions

def raw_dict_to_dataset_conversion(raw_dict):
    stories_pair_conversion_dataset = []
    for story_i in range(0, len(raw_dict)):
        for story_j in range(story_i + 1, len(raw_dict)):
            story_A = raw_dict[story_i]
            story_B = raw_dict[story_j]
            story_A_conversions = get_conversions(story_A)
            story_B_conversions = get_conversions(story_B)
            merged_conversions_A = merge_conversions(story_A_conversions)
            merged_conversions_B = merge_conversions(story_B_conversions)
            for participant_converted_A in merged_conversions_A.keys():
                if len(merged_conversions_A[participant_converted_A]) == 1:
                    conversion_A_not_exist_in_B = True
                    for participant_converted_B in merged_conversions_B.keys():
                        if participant_converted_A == participant_converted_B:
                            conversion_A_not_exist_in_B = False
                    if conversion_A_not_exist_in_B == True and participant_converted_A in story_B["participants"]:
                        if ';' not in participant_converted_A.split() and 'or' not in participant_converted_A.split():
                            story_A_sentences = story_A["sentence_texts"].copy()
                            story_B_sentences = story_B["sentence_texts"].copy()
                            if len(story_A["sentence_texts"]) != len(story_B["sentence_texts"]):
                                if len(story_A["sentence_texts"]) < len(story_B["sentence_texts"]):
                                    for _ in range(len(story_B["sentence_texts"]) - len(story_A["sentence_texts"])):
                                        story_A_sentences.append("")
                                else:
                                    for _ in range(len(story_A["sentence_texts"]) - len(story_B["sentence_texts"])):
                                        story_B_sentences.append("")
                            assert len(story_A_sentences) == len(story_B_sentences)
                            stories_pair_conversion_dataset.append({"story_A_sentences": story_A_sentences, "story_B_sentences": story_B_sentences, "participants": story_A["participants"], "states": story_A["states"], "participant_converted": participant_converted_A, "story_converted": 'A', "conversions": merged_conversions_A[participant_converted_A]})
            for participant_converted_B in merged_conversions_B.keys():
                if len(merged_conversions_B[participant_converted_B]) == 1:
                    conversion_B_not_exist_in_A = True
                    for participant_converted_A in merged_conversions_A.keys():
                        if participant_converted_B == participant_converted_A:
                            conversion_B_not_exist_in_A = False
                    if conversion_B_not_exist_in_A == True and participant_converted_B in story_A["participants"]:
                        if ';' not in participant_converted_B.split() and 'or' not in participant_converted_B.split():
                            story_A_sentences = story_A["sentence_texts"].copy()
                            story_B_sentences = story_B["sentence_texts"].copy()
                            if len(story_A["sentence_texts"]) != len(story_B["sentence_texts"]):
                                if len(story_A["sentence_texts"]) < len(story_B["sentence_texts"]):
                                    for _ in range(len(story_B["sentence_texts"]) - len(story_A["sentence_texts"])):
                                        story_A_sentences.append("")
                                else:
                                    for _ in range(len(story_A["sentence_texts"]) - len(story_B["sentence_texts"])):
                                        story_B_sentences.append("")
                            assert len(story_A_sentences) == len(story_B_sentences)
                            stories_pair_conversion_dataset.append({"story_A_sentences": story_A_sentences, "story_B_sentences": story_B_sentences, "participants": story_B["participants"], "states": story_B["states"], "participant_converted": participant_converted_B, "story_converted": 'B', "conversions": merged_conversions_B[participant_converted_B]})
    return stories_pair_conversion_dataset

def read_propara_raw_data(file_path):
    raw_data = []
    with open(file_path) as f:
        for jsonObj in f:
            example_obj = json.loads(jsonObj)
            raw_data.append(example_obj)
    return raw_data

def story_pair_prompt_generator(story_A_sentences, story_B_sentences):
    """Generate a prompt for information about a pair of stories."""
    prompt = "Story A: " + '\n'
    for sentence_i in range(0, len(story_A_sentences)):
        sentence = story_A_sentences[sentence_i]
        prompt = prompt + str(sentence_i + 1) + ". " + sentence + '\n'
    prompt = prompt + "Story B: " + '\n'
    for sentence_i in range(0, len(story_B_sentences)):
        sentence = story_B_sentences[sentence_i]
        prompt = prompt + str(sentence_i + 1) + ". " + sentence + '\n'
    return prompt

def visualize_conversion_dataset(dataset, num_examples=5):
    for conversion_stories_pair in dataset[:num_examples]:
        print(story_pair_prompt_generator(conversion_stories_pair["story_A_sentences"], conversion_stories_pair["story_B_sentences"]))
        print("Conversion 3rd level: " + conversion_stories_pair["participant_converted"] + " converted in story " + conversion_stories_pair["story_converted"])
        for conversion in conversion_stories_pair["conversions"]:
            print("Conversion 2nd level: " + conversion_stories_pair["participant_converted"] + " converted in sentence " + str(conversion["state_converted_to"]))
            print("Conversion 1st level: " + conversion_stories_pair["participant_converted"] + " converted to " + conversion["participant_converted_to"])
        print()
        print('-' * 100)

def make_conversion_states(dataset):
    for conversion_stories_pair in dataset:
        compact_states = []
        all_participants = conversion_stories_pair["participants"]
        possible_participants_converted_to = [participant for participant in all_participants if participant != conversion_stories_pair["participant_converted"]]
        conversion_stories_pair["possible_participants_converted_to"] = possible_participants_converted_to
        for _ in range(len(possible_participants_converted_to)):
            compact_states.append(0)
        for conversion in conversion_stories_pair["conversions"]:
            participant_converted_to = conversion["participant_converted_to"]
            index_participant_converted_to = possible_participants_converted_to.index(participant_converted_to)
            compact_states[index_participant_converted_to] = 1
        conversion_stories_pair["compact_states"] = compact_states
    return dataset

def get_unique_locations(move_stories_pair):
    unique_locations = []
    for participant_states in move_stories_pair["states"]:
        for participant_state in participant_states:
            if participant_state not in unique_locations:
                unique_locations.append(participant_state)
    return unique_locations

def make_propara_dataset():
    raw_train = read_propara_raw_data("grids.v1.train.json")
    raw_dev = read_propara_raw_data("grids.v1.dev.json")
    raw_test = read_propara_raw_data("grids.v1.test.json")
    propara_dataset = {}
    print("Loading train data...")
    train_conversion, dev_conversion, test_conversion = re_partition_propara_dataset_conversion(raw_train, raw_dev, raw_test)
    propara_dataset["train_conversion"] = make_conversion_states(raw_dict_to_dataset_conversion(train_conversion))
    propara_dataset["dev_conversion"] = make_conversion_states(raw_dict_to_dataset_conversion(dev_conversion))
    propara_dataset["test_conversion"] = make_conversion_states(raw_dict_to_dataset_conversion(test_conversion))

    random.Random(42).shuffle(propara_dataset["train_conversion"])
    random.Random(42).shuffle(propara_dataset["dev_conversion"])
    random.Random(42).shuffle(propara_dataset["test_conversion"])

    print("# stories-pairs in train data (conversion):", len(propara_dataset["train_conversion"]))
    print("# stories-pairs in dev data (conversion):", len(propara_dataset["dev_conversion"]))
    print("# stories-pairs in test data (conversion):", len(propara_dataset["test_conversion"]))

    # print()
    # visualize_conversion_dataset(propara_dataset["test_conversion"], num_examples=10)
    
    return propara_dataset

def re_partition_propara_dataset_conversion(raw_train, raw_dev, raw_test):
    aggregated_dataset = []
    for json_obj in raw_train:
        aggregated_dataset.append(json_obj)
    for json_obj in raw_dev:
        aggregated_dataset.append(json_obj)
    for json_obj in raw_test:
        aggregated_dataset.append(json_obj)
    temp_set, test_set = train_test_split(aggregated_dataset, test_size=0.25, random_state=42)
    train_set, dev_set = train_test_split(temp_set, test_size=0.35, random_state=42)
    return train_set, dev_set, test_set

def write_to_files(propara_dataset):
    if os.path.isfile("train_conversion.json"):
        print("(train_conversion.json exists) Write process suspended. Please delete train_conversion.json and try again.")
    else:
        for json_obj in propara_dataset["train_conversion"]:
            with open("train_conversion.json", "a") as output_file:
                output_file.write(json.dumps(json_obj) + "\n")
    if os.path.isfile("dev_conversion.json"):
        print("(dev_conversion.json exists) Write process suspended. Please delete dev_conversion.json and try again.")
    else:
        for json_obj in propara_dataset["dev_conversion"]:
            with open("dev_conversion.json", "a") as output_file:
                output_file.write(json.dumps(json_obj) + "\n")
    if os.path.isfile("test_conversion.json"):
        print("(test_conversion.json exists) Write process suspended. Please delete test_conversion.json and try again.")
    else:
        for json_obj in propara_dataset["test_conversion"]:
            with open("test_conversion.json", "a") as output_file:
                output_file.write(json.dumps(json_obj) + "\n")

if __name__ == "__main__":
    propara_dataset = make_propara_dataset()
    write_to_files(propara_dataset)