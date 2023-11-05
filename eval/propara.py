import re

# Currently, we only support conversion

EMPTY_RESPONSE = {
    "top_down": {
        "plausibility": {  
            "participant_converted": None,
            "story_converted": None,
        },
        "conflict": {
            "story_converted": None,
            "participant_converted": None,
            "state_converted_to": None,
        },
        "physical_states": {
            "sentence_converted": None,
            "participant_converted": None,
            "participant_converted_to": None,
        },
    },
    "bottom_up": {
        "participant_converted": None,
        "state_converted_to": None,
        "story_converted": None,
        "participant_converted_to": None,
    },
    "plausibility": {
        "participant_converted": None,
        "story_converted": None,
    },
    "conflict": {
        "participant_converted": None,
        "state_converted_to": None,
        "story_converted": None,
    },
    "physical_states": {
        "participant_converted": None,
        "participant_converted_to": None,
    },
}

def response_extractor(generated_text, type="top_down", CoT=False):
    # print("generated text:", generated_text)
    if type not in ["top_down", "bottom_up", "plausibility", "conflict", "physical_states", "conflict_short"]:
        raise ValueError(f"Type {type} not supported")
    top_down_pattern = r"(.*) (?:is|are) converted in story (.*?).\nIn story (.*?), (.*?) (?:is|are) converted in sentence (.*?).\nAfter (.*), (.*?) (?:is|are) converted to (.*)."
    bottom_up_pattern = r"(.*?) (?:is|are) converted to (.*).\nTherefore, (.*?) (?:is|are) converted in sentence (.*?).\nTherefore, (.*) (?:is|are) converted in story (.*)."
    if CoT:
        plausibility_pattern = r"Therefore, (.*?) (?:is|are) converted in story (.*)."
        conflit_pattern = r"Therefore, (.*?) (?:is|are) converted in sentence (.*?) in story (.*)."
        physical_states_pattern = r"Therefore, (.*?) (?:is|are) converted to (.*)."
    else:
        plausibility_pattern = r"(.*?) (?:is|are) converted in story (.*)."
        conflit_pattern = r"(.*?) (?:is|are) converted in sentence (.*?) in story (.*)."
        physical_states_pattern = r"(.*?) (?:is|are) converted to (.*)."
    conflit_pattern_short = r"(.*?) (?:is|are) converted in sentence (.*)."

    top_down_match = re.findall(top_down_pattern, generated_text)
    bottom_up_match = re.findall(bottom_up_pattern, generated_text)
    # print(generated_text)
    plausibility_match = re.findall(plausibility_pattern, generated_text)
    # print("plausibility_match:", plausibility_match)
    conflict_match = re.findall(conflit_pattern, generated_text)
    physical_states_match = re.findall(physical_states_pattern, generated_text)
    conflict_match_short = re.findall(conflit_pattern_short, generated_text)

    # print(top_down_match)
    # print(plausibility_match)
    # print(conflict_match)
    # print(physical_states_match)
    # print(conflict_match_short)

    result = EMPTY_RESPONSE
    top_down = {
        "plausibility": {  
            "participant_converted": None,
            "story_converted": None,
        },
        "conflict": {
            "story_converted": None,
            "participant_converted": None,
            "state_converted_to": None,
        },
        "physical_states": {
            "sentence_converted": None,
            "participant_converted": None,
            "participant_converted_to": None,
        },
    }
    bottom_up = {
        "participant_converted": None,
        "state_converted_to": None,
        "story_converted": None,
        "participant_converted_to": None,
    }
    plausibility = {
        "participant_converted": None,
        "story_converted": None,
    }
    conflict = {
        "participant_converted": None,
        "state_converted_to": None,
        "story_converted": None,
    }
    physical_states = {
        "participant_converted": None,
        "participant_converted_to": None,
    }

    if type == "top_down" and len(top_down_match) > 0:
        plausibility["participant_converted"] = top_down_match[0][0].lower()
        plausibility["story_converted"] = top_down_match[0][1].capitalize()
        conflict["story_converted"] = top_down_match[0][2].capitalize()
        conflict["participant_converted"] = top_down_match[0][3].lower()
        conflict["state_converted_to"] = int(top_down_match[0][4])
        sentence = (
            top_down_match[0][5][0].capitalize() + top_down_match[0][5][1:]
        )
        physical_states["sentence_converted"] = sentence
        physical_states["participant_converted"] = top_down_match[0][6].lower()
        physical_states["participant_converted_to"] = top_down_match[0][7].lower()
        top_down["plausibility"] = plausibility
        top_down["conflict"] = conflict
        top_down["physical_states"] = physical_states
        result["top_down"] = top_down
    
    if type == "bottom_up" and len(bottom_up_match) > 0:
        bottom_up["participant_converted"] = bottom_up_match[0][0].lower()
        bottom_up["participant_converted_to"] = bottom_up_match[0][1].lower()
        bottom_up["state_converted_to"] = int(bottom_up_match[0][3])
        bottom_up["story_converted"] = bottom_up_match[0][5].capitalize()
        result["bottom_up"] = bottom_up

    plausibility = {
        "participant_converted": None,
        "story_converted": None,
    }
    conflict = {
        "participant_converted": None,
        "state_converted_to": None,
        "story_converted": None,
    }
    physical_states = {
        "participant_converted": None,
        "participant_converted_to": None,
    }
    if type == "plausibility" and len(plausibility_match) > 0:
        plausibility["participant_converted"] = plausibility_match[0][0].lower()
        plausibility["story_converted"] = plausibility_match[0][1].capitalize()
    if type == "conflict" and len(conflict_match) > 0:
        conflict["participant_converted"] = conflict_match[0][0].lower()
        conflict["state_converted_to"] = int(conflict_match[0][1])
        conflict["story_converted"] = conflict_match[0][2].capitalize()
    if type == "conflict_short" and len(conflict_match_short) > 0:
        conflict["participant_converted"] = conflict_match_short[0][0].lower()
        conflict["state_converted_to"] = int(conflict_match_short[0][1])
    if type == "physical_states" and len(physical_states_match) > 0:
        physical_states["participant_converted"] = physical_states_match[0][0].lower()
        physical_states["participant_converted_to"] = physical_states_match[0][1].lower()
    result["plausibility"] = plausibility
    result["conflict"] = conflict
    result["physical_states"] = physical_states
    return result

def check_top_down(generated_text, story_pair, CoT=False):
    response = response_extractor(generated_text, type="top_down", CoT=CoT)["top_down"]
    entity = response["plausibility"]["participant_converted"]
    story = response["plausibility"]["story_converted"]
    sentence = response["conflict"]["state_converted_to"]
    converted_to = response["physical_states"]["participant_converted_to"]
    possible_converted_to = story_pair["conversions"][0][
        "participant_converted_to"
    ].lower().split("; ")
    # check if response is consistent
    # if (
    #     response["conflict"]["story_converted"] != story
    #     or response["conflict"]["participant_converted"] != entity
    #     or response["physical_states"]["participant_converted"] != entity
    # ):
    #     print(f"Response is not consistent: {response}")
    #     return 0,0,0
    # check if response is correct
    # if entity != story_pair["participant_converted"].lower():
    #     print(f"Participant converted is wrong. Expected {story_pair['participant_converted']}, got {entity}")
    #     return 0,0,0
    if story != story_pair["story_converted"]:
        # print(f"Story is wrong. Expected {story_pair['story_converted']}, got {story}")
        return 0,0,0
    elif sentence != story_pair["conversions"][0]["state_converted_to"]:
        # print(f"Sentence is wrong. Expected {story_pair['conversions'][0]['state_converted_to']}, got {sentence}")
        return 1,0,0
    elif converted_to not in possible_converted_to:
        # print(f"Participant converted to is wrong. Expected {story_pair['conversions'][0]['participant_converted_to']}, got {converted_to}")
        return 1,1,0
    return 1,1,1


def check_bottom_up(generated_text, story_pair, CoT=False):
    response = response_extractor(generated_text, type="bottom_up", CoT=CoT)["bottom_up"]
    entity = response["participant_converted"]
    converted_to = response["participant_converted_to"]
    possible_converted_to = story_pair["conversions"][0][
        "participant_converted_to"
    ].lower().split("; ")
    sentence = response["state_converted_to"]
    story = response["story_converted"]
    if converted_to not in possible_converted_to:
        print(f"Participant converted to is wrong. Expected {story_pair['conversions'][0]['participant_converted_to']}, got {converted_to}")
        return 0,0,0
    elif sentence != story_pair["conversions"][0]["state_converted_to"]:
        print(f"Sentence is wrong. Expected {story_pair['conversions'][0]['state_converted_to']}, got {sentence}")
        return 1,0,0
    elif story != story_pair["story_converted"]:
        print(f"Story is wrong. Expected {story_pair['story_converted']}, got {story}")
        return 1,1,0
    return 1,1,1

def check_plausibility(generated_text, story_pair, CoT=False):
    # print(generated_text)
    response = response_extractor(generated_text, type="plausibility", CoT=CoT)["plausibility"]
    if response["story_converted"] != None and len(response["story_converted"]) > 0:
        response["story_converted"] = response["story_converted"][0]
    # print(response)
    entity = response["participant_converted"]
    story = response["story_converted"]
    # check if response is correct
    # if entity != story_pair["participant_converted"].lower():
    #     print(f"Participant converted is wrong. Expected {story_pair['participant_converted']}, got {entity}")
    #     return False
    if story != story_pair["story_converted"]:
        # print(f"Story is wrong. Expected {story_pair['story_converted']}, got {story}")
        return False
    return True


def check_conflict(generated_text, story_pair, demo_type="conflict", CoT=False):
    assert demo_type in ["conflict", "conflict_short"]
    response = response_extractor(generated_text, type=demo_type, CoT=CoT)["conflict"]
    if demo_type == "conflict_short":
        story = story_pair["story_converted"]
    else:
        story = response["story_converted"]
    entity = response["participant_converted"]
    sentence = response["state_converted_to"]
    # print(sentence)
    # check if response is correct
    # if story != story_pair["story_converted"]:
    #     print(f"Story is wrong. Expected {story_pair['story_converted']}, got {story}")
    #     return False
    # if entity != story_pair["participant_converted"].lower():
    #     print(f"Participant converted is wrong. Expected {story_pair['participant_converted']}, got {entity}")
    #     return False
    if sentence != story_pair["conversions"][0]["state_converted_to"]:
        # print(f"Sentence is wrong. Expected {story_pair['conversions'][0]['state_converted_to']}, got {sentence}")
        return False
    return True


def check_physical_states(generated_text, story_pair, CoT=False):
    response = response_extractor(generated_text, type="physical_states", CoT=CoT)["physical_states"]
    entity = response["participant_converted"]
    converted_to = response["participant_converted_to"]
    if converted_to != None and converted_to[-1] == '.':
        converted_to = converted_to[:-1]
    possible_converted_to = story_pair["conversions"][0][
        "participant_converted_to"
    ].lower().split("; ")
    # check if response is correct
    # if entity != story_pair["participant_converted"].lower():
    #     print(f"Participant converted is wrong. Expected {story_pair['participant_converted']}, got {entity}")
    #     return False
    if converted_to not in possible_converted_to:
        # print(f"Participant converted to is wrong. Expected {story_pair['conversions'][0]['participant_converted_to']}, got {converted_to}")
        return False
    return True


def check_response(generated_text, story_pair, demo_type="top_down", CoT=False):
    '''
    demo_type can be "top_down", "plausibility", "conflict", "physical_states"
    '''
    assert demo_type in [
        "top_down",
        "bottom_up",
        "plausibility",
        "conflict",
        "physical_states",
        "conflict_short",
    ], f"Demo type {demo_type} not supported"
    if demo_type == "top_down":
        return check_top_down(generated_text, story_pair, CoT=CoT)
    if demo_type == "bottom_up":
        return check_bottom_up(generated_text, story_pair, CoT=CoT)
    if demo_type == "plausibility":
        return check_plausibility(generated_text, story_pair, CoT=CoT)
    if demo_type == "conflict" or demo_type == "conflict_short":
        return check_conflict(generated_text, story_pair, demo_type=demo_type, CoT=CoT)
    if demo_type == "physical_states":
        return check_physical_states(generated_text, story_pair, CoT=CoT)