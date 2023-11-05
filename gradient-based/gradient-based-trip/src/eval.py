import json

from data_utils import read_json_data
import numpy as np

att_default_values = [('h_location', 0), ('conscious', 2), 
                      ('wearing', 0), ('h_wet', 0), 
                      ('hygiene', 0), ('location', 0), 
                      ('exist', 2), ('clean', 0), 
                      ('power', 0), ('functional', 2), 
                      ('pieces', 0), ('wet', 0), 
                      ('open', 0), ('temperature', 0), 
                      ('solid', 0), ('contain', 0), 
                      ('running', 0), ('moveable', 2), 
                      ('mixed', 0), ('edible', 0)]

def find_conflict_indices(length):
    all_pairs = []
    for i in range(length):
        for j in range(i+1, length):
            all_pairs.append([i, j])
    return all_pairs

def convert_output_format_complete_trip(filename, filename1):
    pred_outputs = json.load(open(filename, 'r'))
    original_data = read_json_data(filename1)
    converted_outputs = []
    
    for i in range(0, len(pred_outputs)):
        sample_pred = {}
        entity_collector = [s[0] for s in pred_outputs[i]]
        attribute_collector = np.mean(np.array([s[1] for s in pred_outputs[i]]), axis=0)
        effect_state_collector = np.mean(np.array([s[2] for s in pred_outputs[i]]), axis=0)
        precondition_state_collector = np.mean(np.array([s[3] for s in pred_outputs[i]]), axis=0)
        conflict_collector = np.mean(np.array([s[4] for s in pred_outputs[i]]), axis=0)
        plausible_collector = np.mean(np.array([s[5] for s in pred_outputs[i]]), axis=0)

        all_pairs = find_conflict_indices(len(original_data[i]['stories'][0]["sentences"]))
        sample_pred['story_label'] = original_data[i]['pair_plausible_story']
        sample_pred['conflict_label'] = original_data[i]['pair_confl_pairs'][-1]
        assert len(sample_pred['conflict_label']) == 2

        real_conflicts = []
        for i1 in range(len(original_data[i]['stories'][0]["sentences"])):
            for i2 in range(i1+1, len(original_data[i]['stories'][0]["sentences"])):
                if [i1, i2] == sample_pred['conflict_label']:
                    real_conflicts.append(1)
                else:
                    real_conflicts.append(0)
        
        sample_pred['conflict_label_singular'] = real_conflicts.index(1)

        sample_pred['preconditions_label'] = [[ent[step][0] if step < len(ent) else [0]*20 for step in range(len(original_data[i]['stories'][0]["sentences"]))] for ent in original_data[i]['pair_states']]
        sample_pred['effects_label'] = [[ent[step][1] if step < len(ent) else [0]*20 for step in range(len(original_data[i]['stories'][0]["sentences"]))] for ent in original_data[i]['pair_states']]
        
        sample_pred['story_pred'] = np.argmax(plausible_collector)
        sample_pred['conflict_pred_singular'] = np.argmax(conflict_collector)
        sample_pred['conflict_pred'] = all_pairs[np.argmax(conflict_collector)]
        max_entity_pred_value = float('-inf')
        max_entity_pred_index = -1
        for j, subarray in enumerate(entity_collector):
            if subarray[1] > max_entity_pred_value:
                max_entity_pred_value = subarray[1]
                max_entity_pred_index = j
        confl_entity_pred = max_entity_pred_index
        confl_attribute_pred = np.argmax(attribute_collector)
        confl_effect_state_pred = np.argmax(effect_state_collector)
        confl_precondition_state_pred = np.argmax(precondition_state_collector)
        pair_states_only_relevant = []
        for _ in range(len(original_data[i]["pair_entities"])):
            sentences_state = []
            for _ in range(len(original_data[i]["stories"][0]["sentences"])):
                sentences_state.append([[0] * 20, [0] * 20])
            pair_states_only_relevant.append(sentences_state)
        eff_sentence_idx = sample_pred['conflict_pred'][0] if sample_pred['conflict_pred'][0] < sample_pred['conflict_pred'][1] else sample_pred['conflict_pred'][1]
        pre_sentence_idx = sample_pred['conflict_pred'][1] if sample_pred['conflict_pred'][0] < sample_pred['conflict_pred'][1] else sample_pred['conflict_pred'][0]
        pair_states_only_relevant[confl_entity_pred][eff_sentence_idx][1][confl_attribute_pred] = confl_effect_state_pred
        pair_states_only_relevant[confl_entity_pred][pre_sentence_idx][0][confl_attribute_pred] = confl_precondition_state_pred
        sample_pred['preconditions_pred'] = [[ent[step][0] if step < len(ent) else [0]*20 for step in range(len(original_data[i]['stories'][0]["sentences"]))] for ent in pair_states_only_relevant]
        sample_pred['effects_pred'] = [[ent[step][1] if step < len(ent) else [0]*20 for step in range(len(original_data[i]['stories'][0]["sentences"]))] for ent in pair_states_only_relevant]

        converted_outputs.append(sample_pred)
    return converted_outputs

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

def official_evaluate(filename, original_data_file, all_entities=None):
    if type(filename) == str:
        pred_outputs = json.load(open(filename))
    else:
        pred_outputs = filename
    effects_label = []
    preconditions_label = []
    if type(original_data_file) == str:
        original_data = read_json_data(original_data_file)
    else:
        original_data = original_data_file

    total = 0
    correct = 0
    consistent = 0
    verifiable = 0
    for i in range(0, len(original_data)):
        pred = pred_outputs[int(i)]
        if all_entities:
            ents = all_entities[int(i)][0]
            for ent in ents:
                curr_eid = original_data[i]['pair_entities'].index(ent)
                e = original_data[i]['pair_states'][curr_eid]
                while len(e) < len(original_data[i]['stories'][0]["sentences"]):
                    e.append([[0] * 20, [0] * 20])
                for si, s in enumerate(e):
                    effects_label.append(s[1])
                    preconditions_label.append(s[0])
        else:
            for ei, e in enumerate(original_data[i]['pair_states']):
                while len(e) < len(original_data[i]['stories'][0]["sentences"]):
                    e.append([[0] * 20, [0] * 20])
                for si, s in enumerate(e):
                    effects_label.append(s[1])
                    preconditions_label.append(s[0])

        if pred['story_pred'] == pred['story_label']:
            correct += 1
            if pred['conflict_pred_singular'] == pred['conflict_label_singular']:
                consistent += 1
                states_verifiable = True
                found_states = False
                # Check that effect of first conflict sentence has states which are correct
                for sl, sp in [(pred['effects_label'], pred['effects_pred'])]: # Check preconditions and effects
                    for sl_e, sp_e in zip(sl, sp): # Check all entities
                        for si in [pred['conflict_label'][0]]: # Check conflicting sentences
                            sl_es = sl_e[si]
                            sp_es = sp_e[si]
                            for j, p in enumerate(sp_es): # Check all attributes where there's a nontrivial prediction
                                if p != att_default_values[j][1] and p > 0: # NOTE: p > 0 is required to avoid counting any padding predictions.
                                    found_states = True
                                    if p != sl_es[j]:
                                        states_verifiable = False

                # Check that precondition of second conflict sentence has states which are correct
                #if 'preconditions_label' in pred:
                for sl, sp in [(pred['preconditions_label'], pred['preconditions_pred'])]: # Check preconditions and effects
                    for sl_e, sp_e in zip(sl, sp): # Check all entities        
                        for si in [pred['conflict_label'][1]]: # Check conflicting sentences
                            sl_es = sl_e[si]
                            sp_es = sp_e[si]
                            for j, p in enumerate(sp_es): # Check all attributes where there's a nontrivial prediction
                                if p != att_default_values[j][1] and p > 0: # NOTE: p > 0 is required to avoid counting any padding predictions.
                                    found_states = True
                                    if p != sl_es[j]:
                                        states_verifiable = False
                
                if states_verifiable and found_states:
                    verifiable += 1

        total += 1
    return correct/total, consistent/total, verifiable/total