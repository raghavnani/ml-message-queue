from abc import ABC
from ml_models.abstract_model import MlModel
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from nlp.preprocessing.torch_preprocessing import TorchDataPreprocessing
from nlp.preprocessing.bert_preprocessing import BertPreProcessing
from torch.nn.functional import softmax
import numpy as np


class SentimentModel(MlModel, ABC):

    def __init__(self, path_to_weights):
        MlModel.__init__(self, path_to_weights=path_to_weights)

        self.pytorch_convertor = TorchDataPreprocessing()
        self.bert_preprocessor = BertPreProcessing(max_length=128)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.model = self.instantiate_model()

    def instantiate_model(self):
        sentiment_model = BertForSequenceClassification.from_pretrained(self.path_to_weights, cache_dir=None,
                                                                        num_labels=3)
        sentiment_model.cpu()
        sentiment_model.eval()
        return sentiment_model

    def train_model(self):
        pass

    def predict(self, text):

        ids1, masks1, segments1 = self.bert_preprocessor.get_bert_tokens_masks_segments(text, self.tokenizer)
        ids = self.pytorch_convertor.convert_arrays_to_tensors(ids1)
        masks = self.pytorch_convertor.convert_arrays_to_tensors(masks1)
        segments = self.pytorch_convertor.convert_arrays_to_tensors(segments1)

        logits = self.model(input_ids=ids, token_type_ids=segments, attention_mask=masks)[0]

        logits = softmax(logits, dim=1)

        logits = logits.detach().numpy()

        pred = np.argmax(logits, axis=1)

        if pred == 0:
            sentiment = 'Positive'
        elif pred == 1:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'

        print(sentiment, ','.join(map(str, logits[0])), text)

        return sentiment
