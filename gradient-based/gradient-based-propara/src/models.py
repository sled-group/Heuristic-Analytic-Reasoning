import torch
import torch.nn as nn
import torch.nn.functional as F
from roberta import RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel

class BaselineMixedForConversion(RobertaPreTrainedModel):
    authorized_unexpected_keys = [r"pooler"]
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.state_proj = nn.Linear(config.hidden_size*2, 2)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sent_proj = nn.Linear(config.hidden_size*2, 1)
        self.story_proj = nn.Linear(config.hidden_size*2, 2)
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
        state_label=None, sent_label=None, story_label=None
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
       
        state_loss = None
        sent_loss = None
        story_loss = None

        story_rep = self.story_proj(torch.mean(change_rep_2_stories, dim=1))
        if story_label is not None:
            story_loss = self.loss_fct(story_rep, story_label.unsqueeze(0))

        sent_reps = []
        for i in range(change_rep_2_stories.shape[1]):
            sent_reps.append(change_rep_2_stories[0, i])
        sent_reps = self.sent_proj(torch.stack(sent_reps)).squeeze(-1).unsqueeze(0)
        if sent_label is not None:
            sent_loss = self.loss_fct(sent_reps, sent_label.unsqueeze(0))

        state_rep = self.state_proj(torch.mean(change_rep_2_stories, dim=1))
        if state_label is not None:
            state_loss = self.loss_fct(state_rep, state_label.unsqueeze(0))
        
        return (state_loss, sent_loss, story_loss), (state_rep, sent_reps, story_rep)

class TopDownForConversion(RobertaPreTrainedModel):
    authorized_unexpected_keys = [r"pooler"]
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.state_proj = nn.Linear(config.hidden_size, 2)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sent_proj = nn.Linear(config.hidden_size, 1)
        self.story_proj = nn.Linear(config.hidden_size*2, 2)
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
        state_label=None, sent_label=None, story_label=None
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
       
        state_loss = None
        sent_loss = None
        story_loss = None

        story_rep = self.story_proj(torch.mean(change_rep_2_stories, dim=1))
        if story_label is not None:
            story_loss = self.loss_fct(story_rep, story_label.unsqueeze(0))
        story_pred = torch.argmax(story_rep)

        converted_story_rep = change_reps_raw[story_pred.item()]
        sent_reps = []
        for i in range(converted_story_rep.shape[1]):
            sent_reps.append(converted_story_rep[0, i])
        sent_reps = self.sent_proj(torch.stack(sent_reps)).squeeze(-1).unsqueeze(0)
        if sent_label is not None:
            sent_loss = self.loss_fct(sent_reps, sent_label.unsqueeze(0))
        sent_pred = torch.argmax(sent_reps)

        conversion_rep = converted_story_rep[0][sent_pred.item()].unsqueeze(0)
        state_rep = self.state_proj(conversion_rep)
        if state_label is not None:
            state_loss = self.loss_fct(state_rep, state_label.unsqueeze(0))
        
        return (state_loss, sent_loss, story_loss), (state_rep, sent_reps, story_rep)