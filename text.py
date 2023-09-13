from abc import ABC
from ml_models.abstract_model import MlModel
from transformers import BertTokenizer
from nlp.preprocessing.torch_preprocessing import TorchDataPreprocessing
from nlp.preprocessing.bert_preprocessing import BertPreProcessing
from torch.nn.functional import softmax
import torch
import numpy as np
from torch import nn

class BertForMultiLabelSequenceClassification(nn.Module):

    def __init__(self, embed_model, num_labels=2):
        super(BertForMultiLabelSequenceClassification, self).__init__()

        self.num_labels = num_labels
        self.bert = embed_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)

        # initialize random weight to classifier layer
        nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, input_ids, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids=input_ids,
                                     attention_mask=attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            return loss
        else:
            return torch.sigmoid(logits)

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True


bitcoin_multi_label_model = torch.load('nlp/model_weights/bitcoin/multi-label-both.pt')
bitcoin_multi_label_model.cpu()
bitcoin_multi_label_model.eval()

print(bitcoin_multi_label_model)