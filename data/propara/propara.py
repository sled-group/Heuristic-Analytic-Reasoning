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

def get_moves(story):
    sentences = story["sentence_texts"]
    participants = story["participants"]
    states_for_participants = story["states"]
    moves = []
    for participant_i in range(0, len(participants)):
        for state_step in range(0, len(sentences)): # to avoid state_step + 1 out of index
            proposal_before_move_state = states_for_participants[participant_i][state_step]
            if proposal_before_move_state not in ['-', '?']:
                proposal_after_move_state = states_for_participants[participant_i][state_step + 1]
                if proposal_after_move_state not in ['-', '?']:
                    if proposal_before_move_state != proposal_after_move_state:
                        moves.append({"participant_moved": participants[participant_i], "location_moved_from": proposal_before_move_state, "location_moved_to": proposal_after_move_state, "state_moved_from": state_step, "state_moved_to": state_step + 1})
    return moves

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

def inspect_dataset_moves(raw_dict):
    for story_i in range(0, len(raw_dict)):
        moves = get_moves(raw_dict[story_i])
        if len(moves) > 0:
            print("Story Index", story_i)
            print(make_table(raw_dict[story_i]))
            for move in moves:
                print(move)
            print('-' * 100)

def raw_dict_to_dataset_conversion_for_train_only_one_conversion(raw_dict):
    stories_pair_conversion_dataset = []
    for story_i in range(0, len(raw_dict)):
        for story_j in range(story_i + 1, len(raw_dict)):
            story_A = raw_dict[story_i]
            story_B = raw_dict[story_j]
            if len(story_A["sentence_texts"]) == len(story_B["sentence_texts"]):
                story_A_conversions = get_conversions(story_A)
                if len(story_A_conversions) == 1:
                    story_B_conversions = get_conversions(story_B)
                    if len(story_B_conversions) == 1:
                        conversion_A = story_A_conversions[0]
                        conversion_B = story_B_conversions[0]
                        if conversion_A["participant_converted"] != conversion_B["participant_converted"]:
                            if ';' not in conversion_A["participant_converted"].split() and 'or' not in conversion_A["participant_converted"].split() and ';' not in conversion_B["participant_converted"].split() and 'or' not in conversion_B["participant_converted"].split():
                                stories_pair_conversion_dataset.append({"story_A_sentences": story_A["sentence_texts"], "story_B_sentences": story_B["sentence_texts"], "participants": story_A["participants"], "states": story_A["states"], "participant_converted": conversion_A["participant_converted"], "story_converted": 'A', "conversions": [{"participant_converted_to": conversion_A["participant_converted_to"], "state_converted_from": conversion_A["state_converted_from"], "state_converted_to": conversion_A["state_converted_to"]}]})
                                stories_pair_conversion_dataset.append({"story_A_sentences": story_A["sentence_texts"], "story_B_sentences": story_B["sentence_texts"], "participants": story_B["participants"], "states": story_B["states"], "participant_converted": conversion_B["participant_converted"], "story_converted": 'B', "conversions": [{"participant_converted_to": conversion_B["participant_converted_to"], "state_converted_from": conversion_B["state_converted_from"], "state_converted_to": conversion_B["state_converted_to"]}]})
    return stories_pair_conversion_dataset

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

def raw_dict_to_dataset_move_for_train_only_one_move(raw_dict):
    stories_pair_move_dataset = []
    for story_i in range(0, len(raw_dict)):
        for story_j in range(story_i + 1, len(raw_dict)):
            story_A = raw_dict[story_i]
            story_B = raw_dict[story_j]
            if len(story_A["sentence_texts"]) == len(story_B["sentence_texts"]):
                story_A_moves = get_moves(story_A)
                if len(story_A_moves) == 1:
                    story_B_moves = get_moves(story_B)
                    if len(story_B_moves) == 1:
                        move_A = story_A_moves[0]
                        move_B = story_B_moves[0]
                        if move_A["participant_moved"] != move_B["participant_moved"]:
                            if ';' not in move_A["participant_moved"].split() and 'or' not in move_A["participant_moved"].split() and ';' not in move_B["participant_moved"].split() and 'or' not in move_B["participant_moved"].split():
                                stories_pair_move_dataset.append({"story_A_sentences": story_A["sentence_texts"], "story_B_sentences": story_B["sentence_texts"], "participants": story_A["participants"], "states": story_A["states"], "participant_moved": move_A["participant_moved"], "story_moved": 'A', "moves":[{"location_moved_from": move_A["location_moved_from"], "location_moved_to": move_A["location_moved_to"], "state_moved_from": move_A["state_moved_from"], "state_moved_to": move_A["state_moved_to"]}]})
                                stories_pair_move_dataset.append({"story_A_sentences": story_A["sentence_texts"], "story_B_sentences": story_B["sentence_texts"], "participants": story_B["participants"], "states": story_B["states"], "participant_moved": move_B["participant_moved"], "story_moved": 'B', "moves":[{"location_moved_from": move_B["location_moved_from"], "location_moved_to": move_B["location_moved_to"], "state_moved_from": move_B["state_moved_from"], "state_moved_to": move_B["state_moved_to"]}]})
    return stories_pair_move_dataset

def merge_moves(moves):
    merged_moves = {}
    for move in moves:
        if move["participant_moved"] in merged_moves.keys():
            merged_moves[move["participant_moved"]].append({"location_moved_from": move["location_moved_from"], "location_moved_to": move["location_moved_to"], "state_moved_from": move["state_moved_from"], "state_moved_to": move["state_moved_to"]})
        else:
            merged_moves[move["participant_moved"]] = [{"location_moved_from": move["location_moved_from"], "location_moved_to": move["location_moved_to"], "state_moved_from": move["state_moved_from"], "state_moved_to": move["state_moved_to"]}]
    return merged_moves

def raw_dict_to_dataset_move(raw_dict):
    stories_pair_move_dataset = []
    for story_i in range(0, len(raw_dict)):
        for story_j in range(story_i + 1, len(raw_dict)):
            story_A = raw_dict[story_i]
            story_B = raw_dict[story_j]
            story_A_moves = get_moves(story_A)
            story_B_moves = get_moves(story_B)
            merged_moves_A = merge_moves(story_A_moves)
            merged_moves_B = merge_moves(story_B_moves)
            for participant_moved_A in merged_moves_A.keys():
                if len(merged_moves_A[participant_moved_A]) == 1:
                    move_A_not_exist_in_B = True
                    for participant_moved_B in merged_moves_B.keys():
                        if participant_moved_A == participant_moved_B:
                            move_A_not_exist_in_B = False
                    if move_A_not_exist_in_B == True and participant_moved_A in story_B["participants"]:
                        if ';' not in participant_moved_A.split() and 'or' not in participant_moved_A.split():
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
                            stories_pair_move_dataset.append({"story_A_sentences": story_A_sentences, "story_B_sentences": story_B_sentences, "participants": story_A["participants"], "states": story_A["states"], "participant_moved": participant_moved_A, "story_moved": 'A', "moves": merged_moves_A[participant_moved_A]})
            for participant_moved_B in merged_moves_B.keys():
                if len(merged_moves_B[participant_moved_B]) == 1:
                    move_B_not_exist_in_A = True
                    for participant_moved_A in merged_moves_A.keys():
                        if participant_moved_B == participant_moved_A:
                            move_B_not_exist_in_A = False
                    if move_B_not_exist_in_A == True and participant_moved_B in story_A["participants"]:
                        if ';' not in participant_moved_B.split() and 'or' not in participant_moved_B.split():
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
                            stories_pair_move_dataset.append({"story_A_sentences": story_A_sentences, "story_B_sentences": story_B_sentences, "participants": story_B["participants"], "states": story_B["states"], "participant_moved": participant_moved_B, "story_moved": 'B', "moves": merged_moves_B[participant_moved_B]})
    return stories_pair_move_dataset

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

def visualize_move_dataset(dataset, num_examples=5):
    for move_stories_pair in dataset[:num_examples]:
        print(story_pair_prompt_generator(move_stories_pair["story_A_sentences"], move_stories_pair["story_B_sentences"]))
        print("Move 3rd level: " + move_stories_pair["participant_moved"] + " moved in story " + move_stories_pair["story_moved"])
        for move in move_stories_pair["moves"]:
            print("Move 2nd level: " + move_stories_pair["participant_moved"] + " moved in sentence " + str(move["state_moved_to"]))
            print("Move 1st level: " + move_stories_pair["participant_moved"] + " moved to " + move["location_moved_to"])
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

def make_move_states(dataset):
    for move_stories_pair in dataset:
        compact_states = []
        possible_locations = get_unique_locations(move_stories_pair)
        move_stories_pair["possible_locations"] = possible_locations
        for _ in range(len(possible_locations)):
            compact_states.append(0)
        for move in move_stories_pair["moves"]:
            # location_moved_from = move["location_moved_from"]
            location_moved_to = move["location_moved_to"]
            # index_location_moved_from = possible_locations.index(location_moved_from)
            index_location_moved_to = possible_locations.index(location_moved_to)
            # compact_states[index_location_moved_from] = 1
            compact_states[index_location_moved_to] = 1
        move_stories_pair["compact_states"] = compact_states
    return dataset

def make_propara_dataset():
    raw_train = read_propara_raw_data("grids.v1.train.json")
    raw_dev = read_propara_raw_data("grids.v1.dev.json")
    raw_test = read_propara_raw_data("grids.v1.test.json")
    propara_dataset = {}
    print("Loading train data...")
    train_conversion, dev_conversion, test_conversion = re_partition_propara_dataset_conversion(raw_train, raw_dev, raw_test)
    train_move, dev_move, test_move = re_partition_propara_dataset_move(raw_train, raw_dev, raw_test)
    propara_dataset["train_conversion"] = make_conversion_states(raw_dict_to_dataset_conversion(train_conversion))
    propara_dataset["train_move"] = make_move_states(raw_dict_to_dataset_move(train_move))
    propara_dataset["dev_conversion"] = make_conversion_states(raw_dict_to_dataset_conversion(dev_conversion))
    propara_dataset["dev_move"] = make_move_states(raw_dict_to_dataset_move(dev_move))
    propara_dataset["test_conversion"] = make_conversion_states(raw_dict_to_dataset_conversion(test_conversion))
    propara_dataset["test_move"] = make_move_states(raw_dict_to_dataset_move(test_move))

    random.Random(42).shuffle(propara_dataset["train_conversion"])
    random.Random(42).shuffle(propara_dataset["train_move"])
    random.Random(42).shuffle(propara_dataset["dev_conversion"])
    random.Random(42).shuffle(propara_dataset["dev_move"])
    random.Random(42).shuffle(propara_dataset["test_conversion"])
    random.Random(42).shuffle(propara_dataset["test_move"])

    print("# stories-pairs in train data (conversion):", len(propara_dataset["train_conversion"]))
    print("# stories-pairs in train data (move):", len(propara_dataset["train_move"]))
    print("# stories-pairs in dev data (conversion):", len(propara_dataset["dev_conversion"]))
    print("# stories-pairs in dev data (move):", len(propara_dataset["dev_move"]))
    print("# stories-pairs in test data (conversion):", len(propara_dataset["test_conversion"]))
    print("# stories-pairs in test data (move):", len(propara_dataset["test_move"]))

    # print(propara_dataset["test_move"][0])
    print()
    # visualize_conversion_dataset(propara_dataset["test_conversion"], num_examples=100)
    visualize_move_dataset(propara_dataset["test_move"], num_examples=50)
    for story in test_move:
        if "The water is moving through pipes in the boiler." in story["sentence_texts"]:
            print(make_table(story))
    
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

def re_partition_propara_dataset_move(raw_train, raw_dev, raw_test):
    aggregated_dataset = []
    for json_obj in raw_train:
        aggregated_dataset.append(json_obj)
    for json_obj in raw_dev:
        aggregated_dataset.append(json_obj)
    for json_obj in raw_test:
        aggregated_dataset.append(json_obj)
    temp_set, test_set = train_test_split(aggregated_dataset, test_size=0.23, random_state=42)
    train_set, dev_set = train_test_split(temp_set, test_size=0.4, random_state=42)
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
    if os.path.isfile("train_move.json"):
        print("(train_move.json exists) Write process suspended. Please delete train_move.json and try again.")
    else:
        for json_obj in propara_dataset["train_move"]:
            with open("train_move.json", "a") as output_file:
                output_file.write(json.dumps(json_obj) + "\n")
    if os.path.isfile("dev_move.json"):
        print("(dev_move.json exists) Write process suspended. Please delete dev_move.json and try again.")
    else:
        for json_obj in propara_dataset["dev_move"]:
            with open("dev_move.json", "a") as output_file:
                output_file.write(json.dumps(json_obj) + "\n")
    if os.path.isfile("test_move.json"):
        print("(test_move.json exists) Write process suspended. Please delete test_move.json and try again.")
    else:
        for json_obj in propara_dataset["test_move"]:
            with open("test_move.json", "a") as output_file:
                output_file.write(json.dumps(json_obj) + "\n")


if __name__ == "__main__":
    propara_dataset = make_propara_dataset()
    # write_to_files(propara_dataset)