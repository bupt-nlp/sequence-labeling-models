
import torch
import torch.nn as nn

from transformers.models.bert import BertModel, BertPreTrainedModel, BertConfig
from torchcrf import CRF
from tap import Tap


class JointBertConfig(Tap):
    bert_config = None
    
    intent_size: int = 2
    intent_dropout: float = 0.2

    slot_label_size: int = 4
    slot_dropout: float = 0.5
    use_crf: bool = True
    # scale element for slot loss
    slot_loss_coef: float =0.6
    # ignore `O` slot index
    ignore_index: bool = True

class JointBERT(BertPreTrainedModel):
    """JointBERT: Implementation of [BERT for Joint Intent Classification and Slot Filling](https://arxiv.org/abs/1902.10909)
    """
    def __init__(self, config: JointBertConfig):
        """ init JointBert model with configuration
        """
        super(JointBERT, self).__init__(config.bert_config)
        self.config: JointBertConfig = config

        self.bert: BertModel = BertModel(config=config.bert_config)  # Load pretrained bert

        self.intent_classifier = nn.Sequential(
            nn.Dropout(self.config.intent_dropout),
            nn.Linear(768, self.config.intent_size)
        )

        self.slot_classifier = nn.Sequential(
            nn.Dropout(self.config.intent_dropout),
            nn.Linear(768, self.config.slot_label_size)
        )

        if self.config.use_crf:
            self.crf = CRF(num_tags=self.config.slot_label_size, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids = None, slot_labels_ids = None) -> dict:
        """forward data with the normal data input

        Args:
            input_ids ([type]): the tokenized input ids
            attention_mask ([type]): the attention mask for input id
            token_type_ids ([type]): the segment ids for input
            intent_label_ids ([type]): the tokenized index for intent labels
            slot_labels_ids ([type]): the tokenized index for slot labels

        Returns:
            [type]: the final loss/probs of models
        """
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)

        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)

        total_loss = 0
        # 1. Intent Softmax
        if intent_label_ids is not None:
            if self.config.intent_size == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1))
            else:
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1, self.config.intent_size), intent_label_ids.view(-1))
            total_loss += intent_loss

        # 2. Slot Softmax
        if slot_labels_ids is not None:
            if self.config.use_crf:
                slot_loss = self.crf(slot_logits, slot_labels_ids, mask=attention_mask.byte(), reduction='mean')
                slot_loss = -1 * slot_loss  # negative log-likelihood
            else:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.ignore_index)
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = slot_logits.view(-1, self.config.slot_label_size)[active_loss]
                    active_labels = slot_labels_ids.view(-1)[active_loss]
                    slot_loss = slot_loss_fct(active_logits, active_labels)
                else:
                    slot_loss = slot_loss_fct(slot_logits.view(-1, self.config.slot_label_size), slot_labels_ids.view(-1))
            total_loss += self.config.slot_loss_coef * slot_loss
        return {
            "loss": total_loss,
            "intent_logits": intent_logits,
            "slot_logits": slot_logits
        }