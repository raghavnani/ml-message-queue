import abc
import numpy as np


class BertPreProcessing(metaclass=abc.ABCMeta):

    def __init__(self, max_length: int):

        self.max_seq_length = max_length

    # Mask ids: for every token to mask out tokens used only for the sequence padding
    # (so every sequence has the same length).
    # zero padding
    def get_masks(self, tokens):
        """Mask for padding"""
        if len(tokens) > self.max_seq_length:
            raise IndexError("Token length more than max seq length!")
        return [1] * len(tokens) + [0] * (self.max_seq_length - len(tokens))

    # Segment ids: 0 for one-sentence sequence, 1 if there are two sentences in the sequence and it is the second one
    def get_segments(self, tokens):
        """Segments: 0 for the first sequence, 1 for the second"""
        if len(tokens) > self.max_seq_length:
            raise IndexError("Token length more than max seq length!")
        segments = []
        current_segment_id = 0
        for token in tokens:
            segments.append(current_segment_id)
            if token == "[SEP]":
                current_segment_id = 1
        return segments + [0] * (self.max_seq_length - len(tokens))

    # Token ids: for every token in the sentence. We restore it from the BERT vocab dictionary
    def get_ids(self, tokens, tokenizer):
        """Token ids from Tokenizer vocab"""
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = token_ids + [0] * (self.max_seq_length - len(token_ids))
        return input_ids

    def get_bert_tokens_masks_segments(self, sentence, tokenizer):
        ids = []
        masks = []
        segments = []

        stokens = tokenizer.tokenize(sentence)
        stokens = stokens[0:self.max_seq_length-2]
        stokens = ["[CLS]"] + stokens + ["[SEP]"]

        input_ids = self.get_ids(stokens, tokenizer)
        input_masks = self.get_masks(stokens)
        input_segments = self.get_segments(stokens)

        ids.append(input_ids)
        masks.append(input_masks)
        segments.append(input_segments)

        ids = np.array(ids)
        masks = np.array(masks)
        segments = np.array(segments)

        return (ids, masks, segments)





