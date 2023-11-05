import json
import random

def load_propara_dataset(action_types=["conversion"]):
    """
    Load the ProPara dataset. IMPORTANT: action_types should be a list of strings. The returned dataset has keys action_type. Each action_type has keys train and test. 
    """
    dataset = {}
    for action_type in action_types:
        if action_type not in [
            "conversion",
        ]:
            raise ValueError(f"Action_type {action_type} not supported")
        train_path = f"../data/propara/train_{action_type}.json"
        data_train = []
        with open(train_path, "r") as f:
            for idx, line in enumerate(f):
                datapoint = json.loads(line)
                datapoint["example_id"] = idx
                data_train.append(datapoint)
        test_len = len(data_train)
        test_path = f"../data/propara/test_{action_type}.json"
        data_test = []
        with open(test_path, "r") as f:
            for idx, line in enumerate(f):
                datapoint = json.loads(line)
                datapoint["example_id"] = idx + test_len
                data_test.append(datapoint)
        dataset[action_type] = {"train": data_train, "test": data_test}
    return dataset


def story_prompt_generator(story_pair, story_name):
    """Generate a prompt for information about a single story."""
    if story_name not in ["story_A_sentences", "story_B_sentences"]:
        raise ValueError(f"Story_name {story_name} not supported")
    prompt = "Story: " + "\n"
    for sentence_i in range(0, len(story_pair[story_name])):
        sentence = story_pair[story_name][sentence_i]
        if sentence == "":
            continue
        prompt += str(sentence_i + 1) + ". " + sentence + "\n"
    prompt += f"What happened to {story_pair['participant_converted']}?\n"
    return prompt


def story_pair_prompt_generator(story_pair):
    """Generate a prompt for information about a pair of stories."""
    prompt = "Story A: " + "\n"
    for sentence_i in range(0, len(story_pair["story_A_sentences"])):
        sentence = story_pair["story_A_sentences"][sentence_i]
        if sentence == "":
            continue
        prompt = prompt + str(sentence_i + 1) + ". " + sentence + "\n"
    prompt = prompt + "Story B: " + "\n"
    for sentence_i in range(0, len(story_pair["story_B_sentences"])):
        sentence = story_pair["story_B_sentences"][sentence_i]
        if sentence == "":
            continue
        prompt = prompt + str(sentence_i + 1) + ". " + sentence + "\n"
    prompt += f"What happened to {story_pair['participant_converted']}?\n"
    return prompt


def top_down_demo_full(story_pair, type="conversion"):
    '''type can be "conversion".'''
    if type == "conversion":
        return _conversion_top_down_demo_full(story_pair)
    else:
        raise ValueError(f"Type {type} not supported")


def bottom_up_demo_full(story_pair, type="conversion"):
    '''type can be "conversion".'''
    if type == "conversion":
        return _conversion_bottom_up_demo_full(story_pair)
    else:
        raise ValueError(f"Type {type} not supported")


def plausibility_demo_full(story_pair, type="conversion"):
    '''type can be "conversion".'''
    if type == "conversion":
        return _conversion_plausibility_demo_full(story_pair)
    else:
        raise ValueError(f"Type {type} not supported")


def conflict_demo_full(story_pair, type="conversion", mode="pair"):
    '''type can be "conversion". mode can be "pair" or "single" or "no_story".'''
    if type == "conversion":
        return _conversion_conflict_demo_full(story_pair, mode)
    else:
        raise ValueError(f"Type {type} not supported")


def physical_states_demo_full(story_pair, type="conversion", mode="pair"):
    '''type can be "conversion". mode can be "pair" or "sentence".'''
    if type == "conversion":
        return _conversion_physical_states_demo_full(story_pair, mode)
    else:
        raise ValueError(f"Type {type} not supported")

def _conversion_top_down_demo_full(story_pair):
    '''You shouldn't call this function directly. Use top_down_demo_full instead.'''
    prompt = story_pair_prompt_generator(story_pair)
    entity = story_pair['participant_converted']
    story = story_pair['story_converted'].capitalize()
    sentence = story_pair['conversions'][0]['state_converted_to']
    sentence_text = story_pair[f"story_{story}_sentences"][
        story_pair["conversions"][0]["state_converted_to"] - 1
    ]
    sentence_text = sentence_text[0].lower() + sentence_text[1:-1]
    participant_converted_to = story_pair["conversions"][0][
        "participant_converted_to"
    ].split("; ")
    participant_converted_to = participant_converted_to[
        random.randint(0, len(participant_converted_to) - 1)
    ]
    verb = "are" if is_plural(entity) else "is"
    prompt += f"{entity.capitalize()} {verb} converted in story {story}.\n"
    prompt += f"In story {story}, {entity} {verb} converted in sentence {sentence}.\n"
    prompt += f"After {sentence_text}, {entity} {verb} converted to {participant_converted_to}.\n"
    return prompt


def _conversion_bottom_up_demo_full(story_pair):
    '''You shouldn't call this function directly. Use top_down_demo_full instead.'''
    prompt = story_pair_prompt_generator(story_pair)
    entity = story_pair['participant_converted']
    story = story_pair['story_converted'].capitalize()
    sentence = story_pair['conversions'][0]['state_converted_to']
    sentence_text = story_pair[f"story_{story}_sentences"][
        story_pair["conversions"][0]["state_converted_to"] - 1
    ]
    sentence_text = sentence_text[0].lower() + sentence_text[1:-1]
    participant_converted_to = story_pair["conversions"][0][
        "participant_converted_to"
    ].split("; ")
    participant_converted_to = participant_converted_to[
        random.randint(0, len(participant_converted_to) - 1)
    ]
    verb = "are" if is_plural(entity) else "is"
    prompt += f"{entity.capitalize()} {verb} converted to {participant_converted_to}.\n"
    prompt += f"Therefore, {entity} {verb} converted in sentence {sentence}.\n"
    prompt += f"Therefore, {entity} {verb} converted in story {story}.\n"
    return prompt


def _conversion_plausibility_demo_full(story_pair):
    '''You shouldn't call this function directly. Use plausibility_demo_full instead.'''
    prompt = story_pair_prompt_generator(story_pair)
    entity = story_pair['participant_converted']
    story = story_pair['story_converted'].capitalize()
    verb = "are" if is_plural(entity) else "is"
    prompt += f"{entity.capitalize()} {verb} converted in story {story}.\n"
    return prompt

def _conversion_conflict_demo_full(story_pair, mode="pair"):
    '''You shouldn't call this function directly. Use conflict_demo_full instead.'''
    if mode == "pair":
        prompt = story_pair_prompt_generator(story_pair)
    elif mode in ["single", "no_story"]:
        story = story_pair['story_converted'].capitalize()
        prompt = story_prompt_generator(story_pair, f"story_{story}_sentences")
    else:
        raise ValueError(f"Mode {mode} not supported")
    entity = story_pair['participant_converted']
    sentence = story_pair['conversions'][0]['state_converted_to']
    story = story_pair['story_converted'].capitalize()
    verb = "are" if is_plural(entity) else "is"
    prompt += f"{entity.capitalize()} {verb} converted in sentence {sentence}"
    if mode != "no_story":
        prompt += f" in story {story}.\n"
    else:
        prompt += ".\n"
    return prompt

def _conversion_physical_states_demo_full(story_pair, mode="pair"):
    '''You shouldn't call this function directly. Use physical_states_demo_full instead.'''
    if mode == "pair":
        prompt = story_pair_prompt_generator(story_pair)
    elif mode == "sentence":
        story = story_pair['story_converted'].capitalize()
        sentence = story_pair['conversions'][0]['state_converted_to']
        prompt = story_pair[f"story_{story}_sentences"][sentence - 1] + " "
        prompt += f"What happened to {story_pair['participant_converted']}?\n"
    else:
        raise ValueError(f"Mode {mode} not supported")
    entity = story_pair['participant_converted']
    participant_converted_to = story_pair["conversions"][0][
        "participant_converted_to"
    ].split("; ")
    participant_converted_to = participant_converted_to[
        random.randint(0, len(participant_converted_to) - 1)
    ]
    verb = "are" if is_plural(entity) else "is"
    prompt += f"{entity.capitalize()} {verb} converted to {participant_converted_to}.\n"
    return prompt


def is_plural(word):
    return word[-1] == 's'