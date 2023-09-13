from abc import ABC
from ml_models.abstract_model import MlModel
from transformers import BertTokenizer
from nlp.preprocessing.torch_preprocessing import TorchDataPreprocessing
from nlp.preprocessing.bert_preprocessing import BertPreProcessing
from torch.nn.functional import softmax
import torch
import numpy as np
from torch import nn
from transformers import BertModel


class BitcoinMultiLabelModel(MlModel, ABC):

    def __init__(self, path_to_weights):
        MlModel.__init__(self, path_to_weights=path_to_weights)

        self.pytorch_convertor = TorchDataPreprocessing()
        self.bert_preprocessor = BertPreProcessing(max_length=128)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.model = self.instantiate_model()
        # self.out_labels = np.array([['Crime', 'Regulation', 'Exchanges', 'China', 'Europe', 'Policy', 'Startup', 'Capital',
        #                    'Tech', 'Payment', 'Finance']])

    def instantiate_model(self):

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

        bert_model = BertModel.from_pretrained('nlp/model_weights/fin_model/')
        bitcoin_multi_label_model = BertForMultiLabelSequenceClassification(embed_model=bert_model, num_labels=11)
        bitcoin_multi_label_model.load_state_dict(torch.load('nlp/model_weights/multi-label-both.pt'))
        # bitcoin_multi_label_model = torch.load('nlp/model_weights/bitcoin/multi-label-both.pt')
        bitcoin_multi_label_model.cpu()
        bitcoin_multi_label_model.eval()
        return bitcoin_multi_label_model

    def train_model(self):
        pass

    def predict(self, text):

        ids1, masks1, segments1 = self.bert_preprocessor.get_bert_tokens_masks_segments(text, self.tokenizer)
        ids = self.pytorch_convertor.convert_arrays_to_tensors(ids1)
        masks = self.pytorch_convertor.convert_arrays_to_tensors(masks1)

        logits = self.model(input_ids=ids, attention_mask=masks)

        labels = logits.detach().numpy() > 0.5

        return labels
