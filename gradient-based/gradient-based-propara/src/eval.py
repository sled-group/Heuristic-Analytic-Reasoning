import json

from data_utils import read_json_data
import numpy as np

def convert_output_format_complete_conversion(filename, filename1):
    pred_outputs = json.load(open(filename, 'r'))
    original_data = read_json_data(filename1)
    converted_outputs = []
    
    for i in range(0, len(pred_outputs)):
        sample_pred = {}
        story_outputs = np.mean(np.array([s[2] for s in pred_outputs[i]]), axis=0)
        sent_outputs = np.mean(np.array([s[1] for s in pred_outputs[i]]), axis=0)

        if original_data[i]['story_converted'] == 'A':
            sample_pred['story_label'] = 0
        else:
            sample_pred['story_label'] = 1
        sample_pred['sent_label'] = original_data[i]["conversions"][0]['state_converted_to'] - 1
        sample_pred['states_label'] = original_data[i]['compact_states']
        
        sample_pred['story_pred'] = np.argmax(story_outputs)
        sample_pred['sent_pred'] = np.argmax(sent_outputs)
        sample_pred['states_pred'] = [np.argmax(s[0]) for s in pred_outputs[i]]

        converted_outputs.append(sample_pred)
    return converted_outputs

def official_evaluate(filename, original_data_file):
    if type(filename) == str:
        pred_outputs = json.load(open(filename))
    else:
        pred_outputs = filename
    if type(original_data_file) == str:
        original_data = read_json_data(original_data_file)
    else:
        original_data = original_data_file

    total = 0
    correct = 0
    consistent = 0
    verifiable = 0
    for i in range(0, len(original_data)):
        pred = pred_outputs[i]
        if pred['story_pred'] == pred['story_label']:
            correct += 1
            if pred['sent_pred'] == pred['sent_label']:
                consistent += 1
                if pred['states_pred'] == pred['states_label']:
                    verifiable += 1
        total += 1

    return correct/total, consistent/total, verifiable/total