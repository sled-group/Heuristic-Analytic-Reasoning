import torch
import json 
from collections import Counter
from tqdm import tqdm

def read_json_data(file):
    with open(file, 'r') as f:
        lines = []
        for line in f:
            if line.strip() == '':
                continue
            lines.append(json.loads(str(line)))
    return lines

def make_timestep_sequence(question_tokens, sentence_tokens):
    total_len = 2+len(question_tokens) + sum([len(sent) for sent in sentence_tokens])
    timestep_ids = []
    step0 = [0]*(len(question_tokens)+1) + [2]*(total_len-2-len(question_tokens)) + [0]
    timestep_ids.append(step0)
    for i, sent in enumerate(sentence_tokens):
        this_step = [0]*(len(question_tokens)+1)
        this_step += [1] * sum([len(s) for s in sentence_tokens[:i]])
        this_step += [2] * len(sent) + [3] * sum([len(s) for s in sentence_tokens[i+1:]]) + [0]
        assert len(this_step) == total_len
        timestep_ids.append(this_step)
    return timestep_ids

class ConversionDataset(torch.utils.data.Dataset):

    def __init__(self, data, tokenizer, device, max_train_data):
        super(ConversionDataset, self).__init__()

        if type(data) == str:
            print (data)
            data = read_json_data(data)    
        self.tokenizer = tokenizer
        self.device = device
        self.max_train_data = max_train_data
        self.build_data(data)

    def build_data(self, data):
        self.dataset = []
        story_count = Counter()
        sent_count = Counter()
        for sample in tqdm(data[:self.max_train_data]):
            story_A = sample["story_A_sentences"]
            story_B = sample["story_B_sentences"]
            possible_participants_converted_to = sample['possible_participants_converted_to']
            participant_converted = sample["participant_converted"]
            for entity_id, state in enumerate(sample['compact_states']):
                question = participant_converted + "?!</s>" + possible_participants_converted_to[entity_id] + "?!</s>"
                question_tokens = self.tokenizer.tokenize(question.lower())
                sentence_tokens_A = [self.tokenizer.tokenize(sent.lower(), add_prefix_space=True) for sent in story_A]
                sentence_tokens_B = [self.tokenizer.tokenize(sent.lower(), add_prefix_space=True) for sent in story_B]
                para_tokens_A = [w for w in question_tokens]
                para_tokens_B = [w for w in question_tokens]
                for sent in sentence_tokens_A:
                    para_tokens_A += sent
                for sent in sentence_tokens_B:
                    para_tokens_B += sent
                para_ids_A = [self.tokenizer.cls_token_id] + self.tokenizer.convert_tokens_to_ids(para_tokens_A) + [self.tokenizer.sep_token_id]
                para_ids_B = [self.tokenizer.cls_token_id] + self.tokenizer.convert_tokens_to_ids(para_tokens_B) + [self.tokenizer.sep_token_id]
                timestep_ids_A = make_timestep_sequence(question_tokens, sentence_tokens_A)[1:]
                timestep_ids_B = make_timestep_sequence(question_tokens, sentence_tokens_B)[1:]
                story_label = None
                if sample["story_converted"] == 'A':
                    story_label = 0
                else:
                    story_label = 1
                sent_label = sample["conversions"][0]["state_converted_to"] - 1
                state_label = state
                story_count[story_label] += 1
                sent_count[sent_label] += 1
                exp = {'input_ids_A': [para_ids_A]*len(timestep_ids_A), 'attention_mask_A': [1]*len(para_ids_A), 'timestep_type_ids_A': timestep_ids_A, 'input_ids_B': [para_ids_B]*len(timestep_ids_B), 'attention_mask_B': [1]*len(para_ids_B), 'timestep_type_ids_B': timestep_ids_B}
                exp['sent_label'] = sent_label
                exp['story_label'] = story_label
                exp['state_label'] = state_label
                self.dataset.append(exp)
        print (len(self.dataset))
        print (story_count)
        print (sent_count)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):

        instance = self.dataset[index]
        return instance

    def collate(self, batch):

        batch = batch[0]
        input_ids_A = torch.tensor(batch['input_ids_A'], device=self.device)
        att_mask_A = torch.tensor([batch['attention_mask_A'] for _ in range(len(batch['input_ids_A']))], device=self.device)
        timestep_A = torch.tensor(batch['timestep_type_ids_A'], device=self.device)
        input_ids_B = torch.tensor(batch['input_ids_B'], device=self.device)
        att_mask_B = torch.tensor([batch['attention_mask_B'] for _ in range(len(batch['input_ids_B']))], device=self.device)
        timestep_B = torch.tensor(batch['timestep_type_ids_B'], device=self.device)
        state_label = torch.tensor(batch['state_label'], device=self.device)
        sent_label = torch.tensor(batch['sent_label'], device=self.device)
        story_label = torch.tensor(batch['story_label'], device=self.device)
        
        return {'input_ids_A': input_ids_A, 'input_ids_B': input_ids_B,
                'attention_mask_A': att_mask_A, 'attention_mask_B': att_mask_B,
                'timestep_type_ids_A': timestep_A, 'timestep_type_ids_B': timestep_B,
                'state_label': state_label, 'sent_label': sent_label, 'story_label': story_label}