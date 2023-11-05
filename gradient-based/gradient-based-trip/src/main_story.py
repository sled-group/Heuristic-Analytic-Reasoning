import os
import torch
import torch.nn.functional as F
import numpy as np
import random
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, AdamW, get_linear_schedule_with_warmup
from data_utils import read_json_data, make_timestep_sequence, TripDataset
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
import argparse
from models import BaselineMixedForTrip, TopDownForTrip
from tqdm import tqdm
import json
from eval import official_evaluate, convert_output_format_complete_trip

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)    

def test_model_trip(args, model, test_set, name="test_full_trip", iteration=0):
    all_preds = []
    for sample in tqdm(test_set):
        story_A = sample['stories'][0]["sentences"]
        story_B = sample['stories'][1]["sentences"]
        participants = sample['pair_entities']
        sample_pred = []
        for entity_id, states in enumerate(sample["pair_states"]):
            question = participants[entity_id] + "?!</s>"
            question_tokens = roberta_tokenizer.tokenize(question.lower())
            sentence_tokens_A = [roberta_tokenizer.tokenize(sent.lower(), add_prefix_space=True) for sent in story_A]
            sentence_tokens_B = [roberta_tokenizer.tokenize(sent.lower(), add_prefix_space=True) for sent in story_B]
            para_tokens_A = [w for w in question_tokens]
            para_tokens_B = [w for w in question_tokens]
            for sent in sentence_tokens_A:
                para_tokens_A += sent
            for sent in sentence_tokens_B:
                para_tokens_B += sent
            para_ids_A = [roberta_tokenizer.cls_token_id] + roberta_tokenizer.convert_tokens_to_ids(para_tokens_A) + [roberta_tokenizer.sep_token_id]
            para_ids_B = [roberta_tokenizer.cls_token_id] + roberta_tokenizer.convert_tokens_to_ids(para_tokens_B) + [roberta_tokenizer.sep_token_id]
            timestep_ids_A = make_timestep_sequence(question_tokens, sentence_tokens_A)[1:]
            timestep_ids_B = make_timestep_sequence(question_tokens, sentence_tokens_B)[1:]

            batch_input_A = torch.tensor([para_ids_A]*len(timestep_ids_A), device=args.device)
            timestep_ids_A = torch.tensor(timestep_ids_A, device=args.device)
            batch_input_B = torch.tensor([para_ids_B]*len(timestep_ids_B), device=args.device)
            timestep_ids_B = torch.tensor(timestep_ids_B, device=args.device)
            
            _, outputs = model(input_ids_A=batch_input_A, input_ids_B=batch_input_B, timestep_type_ids_A=timestep_ids_A, timestep_type_ids_B=timestep_ids_B)
            sample_pred.append([F.softmax(outputs[0], dim=1)[0].tolist(), F.softmax(outputs[1], dim=1)[0].tolist(), F.softmax(outputs[2], dim=1)[0].tolist(), F.softmax(outputs[3], dim=1)[0].tolist(), F.softmax(outputs[4], dim=1)[0].tolist(), F.softmax(outputs[5], dim=1)[0].tolist()])   
        all_preds.append(sample_pred)
    with open(f"{args.output_dir}/{name}_{iteration}_output.json", 'w') as fout:
        json.dump(all_preds, fout)            
    print(f"The {name} iteration ", str(iteration), " final results are: ")
    converted_results = convert_output_format_complete_trip(f"{args.output_dir}/{name}_{iteration}_output.json", f"../data/Trip/{name}.json")
    accuracy, consistency, verifiability = official_evaluate(converted_results, f"../data/Trip/{name}.json")
    print({"accuracy": accuracy, "consistency": consistency, "verifiability": verifiability})

    return verifiability, consistency, accuracy

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(args, model):
    train_set = TripDataset('../data/Trip/train_trip.json', tokenizer=roberta_tokenizer, device=args.device, max_train_data=int(args.max_train_data))
    full_dev_set = read_json_data('../data/Trip/dev_full_trip.json')
    curated_dev_set = read_json_data('../data/Trip/dev_curated_trip.json')

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-6)
    t_total = len(train_set) // args.gradient_accumulation_steps * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*t_total), num_training_steps=t_total)

    best_verifiability_for_full_dev = -1
    best_verifiability_for_curated_dev = -1

    global_step = 0

    train_batch = DataLoader(dataset=train_set, batch_size=args.per_gpu_train_batch_size, shuffle=True, collate_fn=train_set.collate)
    for iteration in tqdm(range(args.num_train_epochs)):

        if iteration == 0:
            model.eval()
            with torch.no_grad():
                verifiability_full, consistency_full, accuracy_full = test_model_trip(args, model, full_dev_set, name="dev_full_trip", iteration=0)
            results_epoch_full_dev_json = {"accuracy": accuracy_full, "consistency": consistency_full, "verifiability": verifiability_full}
            with open(f'{args.output_dir}/results_epoch_full_dev.json', "a") as output_file:
                output_file.write(json.dumps(results_epoch_full_dev_json) + "\n")
            
            with torch.no_grad():
                verifiability_curated, consistency_curated, accuracy_curated = test_model_trip(args, model, curated_dev_set, name="dev_curated_trip", iteration=0)
            results_epoch_curated_dev_json = {"accuracy": accuracy_curated, "consistency": consistency_curated, "verifiability": verifiability_curated}
            with open(f'{args.output_dir}/results_epoch_curated_dev.json', "a") as output_file:
                output_file.write(json.dumps(results_epoch_curated_dev_json) + "\n")
        
        it_total_loss = 0
        model.train()
        for batch in tqdm(train_batch):
            total_loss = 0
            with autocast():
                losses, outputs = model(**batch)

            if losses[0] is not None:
                total_loss += losses[0]

            if losses[1] is not None:
                total_loss += losses[1]
            
            if losses[2] is not None:
                total_loss += losses[2]

            if args.gradient_accumulation_steps > 1:
                total_loss = total_loss / args.gradient_accumulation_steps
            total_loss.backward()

            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            it_total_loss += total_loss.item()
            global_step += 1
           
        print("The iteration loss is: ", it_total_loss)
            
        model.eval()

        with torch.no_grad():
            verifiability_full, consistency_full, accuracy_full = test_model_trip(args, model, full_dev_set, name="dev_full_trip", iteration=iteration+1)
        results_epoch_full_dev_json = {"accuracy": accuracy_full, "consistency": consistency_full, "verifiability": verifiability_full}
        with open(f'{args.output_dir}/results_epoch_full_dev.json', "a") as output_file:
            output_file.write(json.dumps(results_epoch_full_dev_json) + "\n")
      
        if verifiability_full >= best_verifiability_for_full_dev:
            torch.save(model.state_dict(), f"{args.output_dir}/best_model_for_full_dev")
            best_verifiability_for_full_dev = verifiability_full

        with torch.no_grad():
            verifiability_curated, consistency_curated, accuracy_curated = test_model_trip(args, model, curated_dev_set, name="dev_curated_trip", iteration=iteration+1)
        results_epoch_curated_dev_json = {"accuracy": accuracy_curated, "consistency": consistency_curated, "verifiability": verifiability_curated}
        with open(f'{args.output_dir}/results_epoch_curated_dev.json', "a") as output_file:
            output_file.write(json.dumps(results_epoch_curated_dev_json) + "\n")
      
        if verifiability_curated >= best_verifiability_for_curated_dev:
            torch.save(model.state_dict(), f"{args.output_dir}/best_model_for_curated_dev")
            best_verifiability_for_curated_dev = verifiability_curated

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", default="output", type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    
    parser.add_argument("--dataset_name", default="trip", type=str, choices=["trip"],
                        help="Dataset choice")
    
    parser.add_argument("--model", default="top-down", type=str, choices=["baseline-mixed", "top-down"],
                        help="Model choice")
    
    parser.add_argument("--max_train_data", default="799", type=str, choices=["799", "5"],
                        help="Max available training data: Full Trip Training Dataset: 799, Debugging Trip Training Dataset: 5")
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--per_gpu_train_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for training.")
    
    parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    
    parser.add_argument("--learning_rate", default=5e-6, type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    
    parser.add_argument("--num_train_epochs", default=10, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    args = parser.parse_args()

    new_out_dir = f"{args.output_dir}/{args.dataset_name}_{args.model}_epochs_{args.num_train_epochs}_lr_{args.learning_rate}_seed_{args.seed}"
    if args.do_train or (not args.do_train and os.path.exists(new_out_dir)):
        args.output_dir = new_out_dir

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        print ("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    args.n_gpu = 1
    args.device = device = 'cuda:0'
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.do_train:
        os.system("cp main_story.py %s" % os.path.join(args.output_dir, 'main_story.py'))
        os.system("cp data_utils.py %s" % os.path.join(args.output_dir, 'data_utils.py'))
        os.system("cp models.py %s" % os.path.join(args.output_dir, 'models.py'))
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
    
    set_seed(args)
    if args.model == "baseline-mixed":
        model = BaselineMixedForTrip.from_pretrained('roberta-large', return_dict=True)
    elif args.model == "top-down":
        model = TopDownForTrip.from_pretrained('roberta-large', return_dict=True)
    print (count_parameters(model))

    model.to(device)
    if args.do_train:
        train(args, model)
    if args.do_eval:
        model.load_state_dict(torch.load(f"{args.output_dir}/best_model_for_full_dev"))
        model.to(device)

        full_test_set = read_json_data(f'../data/Trip/test_full_trip.json')
        verifiability_full, consistency_full, accuracy_full = test_model_trip(args, model, full_test_set, name="test_full_trip", iteration=-1)
        final_full_test_results_json = {"accuracy": accuracy_full, "consistency": consistency_full, "verifiability": verifiability_full}
        with open(f'{args.output_dir}/final_full_test_results.json', "a") as output_file:
            output_file.write(json.dumps(final_full_test_results_json) + "\n")

        model.load_state_dict(torch.load(f"{args.output_dir}/best_model_for_full_dev"))
        model.to(device)
        
        curated_test_set = read_json_data(f'../data/Trip/test_curated_trip.json')
        verifiability_curated, consistency_curated, accuracy_curated = test_model_trip(args, model, curated_test_set, name="test_curated_trip", iteration=-1)
        final_curated_test_results_json = {"accuracy": accuracy_curated, "consistency": consistency_curated, "verifiability": verifiability_curated}
        with open(f'{args.output_dir}/final_curated_test_results.json', "a") as output_file:
            output_file.write(json.dumps(final_curated_test_results_json) + "\n")

        model.load_state_dict(torch.load(f"{args.output_dir}/best_model_for_full_dev"))
        model.to(device)
        
        curated_test_set = read_json_data(f'../data/Trip/test_implicit_trip.json')
        verifiability_curated, consistency_curated, accuracy_curated = test_model_trip(args, model, curated_test_set, name="test_implicit_trip", iteration=-1)
        final_curated_test_results_json = {"accuracy": accuracy_curated, "consistency": consistency_curated, "verifiability": verifiability_curated}
        with open(f'{args.output_dir}/final_implicit_test_results.json', "a") as output_file:
            output_file.write(json.dumps(final_curated_test_results_json) + "\n")
if __name__ == '__main__':
    main()