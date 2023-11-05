import torch
import torch.nn as nn
import torch.nn.functional as F
from roberta import RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel

def find_conflict_indices(length):
    all_pairs = []
    for i in range(length):
        for j in range(i+1, length):
            all_pairs.append([i, j])
    return all_pairs

class BaselineMixedForTrip(RobertaPreTrainedModel):
    authorized_unexpected_keys = [r"pooler"]
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.entity_proj = nn.Linear(config.hidden_size*2, 2)
        self.attribute_proj = nn.Linear(config.hidden_size*2, 20)
        self.effect_state_proj = nn.Linear(config.hidden_size*2, 3)
        self.precondition_state_proj = nn.Linear(config.hidden_size*2, 3)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.conflict_proj = nn.Linear(config.hidden_size*4, 1)
        self.plausible_proj = nn.Linear(config.hidden_size*2, 2)
        self.loss_fct = nn.CrossEntropyLoss()

        self.init_weights()

    def forward(
        self,
        input_ids_A=None,
        input_ids_B=None,
        attention_mask_A=None,
        attention_mask_B=None,
        token_type_ids=None,
        timestep_type_ids_A=None,
        timestep_type_ids_B=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        confl_entity_label=None, confl_attribute_label=None, confl_effect_state_label=None, confl_precondition_state_label=None, conflict_label=None, plausible_label=None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs_A = self.roberta(
            input_ids_A,
            attention_mask=attention_mask_A,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            timestep_type_ids=timestep_type_ids_A,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        outputs_B = self.roberta(
            input_ids_B,
            attention_mask=attention_mask_B,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            timestep_type_ids=timestep_type_ids_B,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output_A = outputs_A[0]
        cls_token_A = sequence_output_A[:, 0, :]
        cls_token_A = cls_token_A.unsqueeze(0)

        sequence_output_B = outputs_B[0]
        cls_token_B = sequence_output_B[:, 0, :]
        cls_token_B = cls_token_B.unsqueeze(0)
       
        change_rep_A = self.dropout(cls_token_A)
        change_rep_B = self.dropout(cls_token_B)

        change_reps_raw = (change_rep_A, change_rep_B)

        change_rep_2_stories = torch.cat(change_reps_raw, 2)
       
        change_loss = None
        conflict_loss = None
        plausible_loss = None
        
        plausible_rep = self.plausible_proj(torch.mean(change_rep_2_stories, dim=1))
        if plausible_label is not None:
            plausible_loss = self.loss_fct(plausible_rep, plausible_label.unsqueeze(0))
        
        conflict_reps = []
        for i in range(change_rep_2_stories.shape[1]):
            for j in range(i+1, change_rep_2_stories.shape[1]):
                conflict_reps.append(torch.cat([change_rep_2_stories[0, i], change_rep_2_stories[0, j]], dim=-1))
        conflict_reps = self.conflict_proj(torch.stack(conflict_reps)).squeeze(-1).unsqueeze(0)
        if conflict_label is not None:
            conflict_loss = self.loss_fct(conflict_reps, conflict_label.unsqueeze(0))
        
        confl_entity_rep = self.entity_proj(torch.mean(change_rep_2_stories, dim=1))
        confl_attribute_rep = self.attribute_proj(torch.mean(change_rep_2_stories, dim=1))
        confl_effect_state_rep = self.effect_state_proj(torch.mean(change_rep_2_stories, dim=1))
        confl_precondition_state_rep = self.precondition_state_proj(torch.mean(change_rep_2_stories, dim=1))

        states_loss_will_backward = True
        if confl_attribute_label is not None:
            if confl_attribute_label.item() == -1: 
                states_loss_will_backward = False
        if states_loss_will_backward and confl_entity_label is not None and confl_attribute_label is not None and confl_effect_state_label is not None and confl_precondition_state_label is not None:
            entity_loss = self.loss_fct(confl_entity_rep, confl_entity_label.unsqueeze(0))
            attribute_loss = self.loss_fct(confl_attribute_rep, confl_attribute_label.unsqueeze(0))
            effect_state_loss = self.loss_fct(confl_effect_state_rep, confl_effect_state_label.unsqueeze(0))
            precondition_state_loss = self.loss_fct(confl_precondition_state_rep, confl_precondition_state_label.unsqueeze(0))
            change_loss = (entity_loss + attribute_loss + effect_state_loss + precondition_state_loss) / 4
        
        return (change_loss, conflict_loss, plausible_loss), (confl_entity_rep, confl_attribute_rep, confl_effect_state_rep, confl_precondition_state_rep, conflict_reps, plausible_rep)

class TopDownForTrip(RobertaPreTrainedModel):
    authorized_unexpected_keys = [r"pooler"]
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.entity_proj = nn.Linear(config.hidden_size*2, 2)
        self.attribute_proj = nn.Linear(config.hidden_size*2, 20)
        self.effect_state_proj = nn.Linear(config.hidden_size, 3)
        self.precondition_state_proj = nn.Linear(config.hidden_size, 3)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.conflict_proj = nn.Linear(config.hidden_size*2, 1)
        self.plausible_proj = nn.Linear(config.hidden_size*2, 2)
        self.loss_fct = nn.CrossEntropyLoss()

        self.init_weights()

    def forward(
        self,
        input_ids_A=None,
        input_ids_B=None,
        attention_mask_A=None,
        attention_mask_B=None,
        token_type_ids=None,
        timestep_type_ids_A=None,
        timestep_type_ids_B=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        confl_entity_label=None, confl_attribute_label=None, confl_effect_state_label=None, confl_precondition_state_label=None, conflict_label=None, plausible_label=None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs_A = self.roberta(
            input_ids_A,
            attention_mask=attention_mask_A,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            timestep_type_ids=timestep_type_ids_A,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        outputs_B = self.roberta(
            input_ids_B,
            attention_mask=attention_mask_B,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            timestep_type_ids=timestep_type_ids_B,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output_A = outputs_A[0]
        cls_token_A = sequence_output_A[:, 0, :]
        cls_token_A = cls_token_A.unsqueeze(0)

        sequence_output_B = outputs_B[0]
        cls_token_B = sequence_output_B[:, 0, :]
        cls_token_B = cls_token_B.unsqueeze(0)
       
        change_rep_A = self.dropout(cls_token_A)
        change_rep_B = self.dropout(cls_token_B)

        change_reps_raw = (change_rep_A, change_rep_B)

        change_rep_2_stories = torch.cat(change_reps_raw, 2)
       
        change_loss = None
        conflict_loss = None
        plausible_loss = None
        
        plausible_rep = self.plausible_proj(torch.mean(change_rep_2_stories, dim=1))
        if plausible_label is not None:
            plausible_loss = self.loss_fct(plausible_rep, plausible_label.unsqueeze(0))
        plausible_pred = torch.argmax(plausible_rep)

        implausible_story_rep = change_reps_raw[1 - plausible_pred.item()] #if pred=1, index 0 is implausible; if pred=0, index 1 is implausible
        conflict_reps = []
        for i in range(implausible_story_rep.shape[1]):
            for j in range(i+1, implausible_story_rep.shape[1]):
                conflict_reps.append(torch.cat([implausible_story_rep[0, i], implausible_story_rep[0, j]], dim=-1))
        conflict_reps = self.conflict_proj(torch.stack(conflict_reps)).squeeze(-1).unsqueeze(0)
        if conflict_label is not None:
            conflict_loss = self.loss_fct(conflict_reps, conflict_label.unsqueeze(0))

        all_pairs = find_conflict_indices(implausible_story_rep.size()[1])
        conflict_pred = all_pairs[torch.argmax(conflict_reps).item()]
        
        confl_sent_effect = implausible_story_rep[0, conflict_pred[0]]
        confl_sent_precondition = implausible_story_rep[0, conflict_pred[1]]
        confl_sent = torch.cat((confl_sent_precondition, confl_sent_effect), dim=-1)
        confl_entity_rep = self.entity_proj(confl_sent).unsqueeze(0)
        confl_attribute_rep = self.attribute_proj(confl_sent).unsqueeze(0)
        confl_effect_state_rep = self.effect_state_proj(confl_sent_effect).unsqueeze(0)
        confl_precondition_state_rep = self.precondition_state_proj(confl_sent_precondition).unsqueeze(0)

        states_loss_will_backward = True
        if confl_attribute_label is not None:
            if confl_attribute_label.item() == -1: 
                states_loss_will_backward = False
        if states_loss_will_backward and confl_entity_label is not None and confl_attribute_label is not None and confl_effect_state_label is not None and confl_precondition_state_label is not None:
            entity_loss = self.loss_fct(confl_entity_rep, confl_entity_label.unsqueeze(0))
            attribute_loss = self.loss_fct(confl_attribute_rep, confl_attribute_label.unsqueeze(0))
            effect_state_loss = self.loss_fct(confl_effect_state_rep, confl_effect_state_label.unsqueeze(0))
            precondition_state_loss = self.loss_fct(confl_precondition_state_rep, confl_precondition_state_label.unsqueeze(0))
            change_loss = (entity_loss + attribute_loss + effect_state_loss + precondition_state_loss) / 4
        
        return (change_loss, conflict_loss, plausible_loss), (confl_entity_rep, confl_attribute_rep, confl_effect_state_rep, confl_precondition_state_rep, conflict_reps, plausible_rep)