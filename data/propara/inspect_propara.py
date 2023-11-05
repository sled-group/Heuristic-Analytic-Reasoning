import json
from tabulate import tabulate

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
    
def visualize_conversion_dataset(dataset, original_dataset, num_examples=5):
    for conversion_stories_pair in dataset[:num_examples]:
        for story in original_dataset:
            reduced_story_A_sentences = conversion_stories_pair["story_A_sentences"]
            reduced_story_B_sentences = conversion_stories_pair["story_B_sentences"]
            while "" in reduced_story_A_sentences:
                reduced_story_A_sentences.remove("")
            while "" in reduced_story_B_sentences:
                reduced_story_B_sentences.remove("")
            if story["sentence_texts"] == reduced_story_A_sentences:
                print(make_table(story))
            if story["sentence_texts"] == reduced_story_B_sentences:
                print(make_table(story))
        print(story_pair_prompt_generator(conversion_stories_pair["story_A_sentences"], conversion_stories_pair["story_B_sentences"]))
        print("Conversion 3rd level: " + conversion_stories_pair["participant_converted"] + " converted in story " + conversion_stories_pair["story_converted"])
        for conversion in conversion_stories_pair["conversions"]:
            print("Conversion 2nd level: " + conversion_stories_pair["participant_converted"] + " converted in sentence " + str(conversion["state_converted_to"]))
            print("Conversion 1st level: " + conversion_stories_pair["participant_converted"] + " converted to " + conversion["participant_converted_to"])
        print()
        print('-' * 100)
        
def read_propara_data(file_path):
    data = []
    with open(file_path) as f:
        for jsonObj in f:
            example_obj = json.loads(jsonObj)
            data.append(example_obj)
    return data

propara_conversion = read_propara_data("test_conversion.json")

original_propara = read_propara_data("grids.v1.test.json")
original_propara.extend(read_propara_data("grids.v1.dev.json"))
original_propara.extend(read_propara_data("grids.v1.train.json"))

# print(visualize_conversion_dataset(propara_conversion, original_propara, num_examples=5))

for story in original_propara:
    if "The heat is used to turn water into steam." in story["sentence_texts"]:
        print(make_table(story))

print(len(propara_conversion))

# for i in range(0, 213):
#     story = propara_conversion[i]
#     if "Water is exposed to heat energy, like sunlight." in story["story_A_sentences"]:
#         print(story)
#         print()


