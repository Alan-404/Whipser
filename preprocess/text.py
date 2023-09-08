import re
import pickle
import os
import numpy as np

class Cleaner:
    def __init__(self, puncs: list = r"([:./,?!@#$%^&=`~*\(\)\[\]\"\'\-\\])") -> None:
        self.puncs = puncs
    def clean(self, seq: str):
        seq = re.sub(self.puncs, r" \1 ", seq)
        seq = seq.strip()
        seq = re.sub("\s\s+", " ", seq)
        seq = seq.lower()
        return seq

class TextProcessor:
    def __init__(self, tokenizer_path: str = None) -> None:
        self.cleaner = Cleaner()
        self.dictionary = []
        self.tokenizer_path = tokenizer_path
        if tokenizer_path is not None and os.path.exists(tokenizer_path):
            self.load_tokenizer(tokenizer_path)
        else:
            self.dictionary.append("<pad>")

    def load_tokenizer(self, path: str):
        with open(path, 'rb') as file:
            self.dictionary = pickle.load(file)

    def save_tokenizer(self, path: str):
        with open(path, 'wb') as file:
            pickle.dump(self.dictionary, file, protocol=pickle.HIGHEST_PROTOCOL)

    def add_dictionary(self, token: str):
        self.dictionary.append(token)

    def text_to_sequences(self, sequence: str, training: bool):
        tokens = sequence.split(" ")
        digit = []
        for token in tokens:
            digit.append(self.tokenize(token, training))
        return np.array(digit)

    def tokenize(self, token: str, training: bool = True):
        if token in self.dictionary:
            if token == "<unk>":
                return self.dictionary.index("<oov>")
            return self.dictionary.index(token)
        else:
            if training == False:
                return self.dictionary.index("<oov>")
            else:
                self.add_dictionary(token)
                return self.dictionary.index(token)

    def process(self, sequences: list[str], maxlen: int = None, training: bool = True):
        max_length = 0
        digits = []
        for index, sequence in enumerate(sequences):
            text = self.cleaner.clean(sequence)
            digit = self.text_to_sequences(text, training)
            digits.append(digit)
            if max_length < len(digit):
                max_length = len(digit)

        if maxlen is not None:
            max_length = maxlen
        if self.tokenizer_path:
            self.save_tokenizer(self.tokenizer_path)
        digits = self.pad_sequences(digits, max_length)

        return digits
    
    def padding_sequence(self, sequence: np.ndarray, padding: str, maxlen: int) -> np.ndarray:
        delta = maxlen - len(sequence)
        zeros = np.zeros(delta, dtype=np.int64)

        if padding.strip().lower() == 'post':
            return np.concatenate((sequence, zeros), axis=0)
        elif padding.strip().lower() == 'pre':
            return np.concatenate((zeros, sequence), axis=0)

    def truncating_sequence(self, sequence, truncating: str, maxlen: int) -> np.ndarray:
        if truncating.strip().lower() == 'post':
            return sequence[0:maxlen]
        elif truncating.strip().lower() == 'pre':
            delta = sequence.shape[0] - maxlen
            return sequence[delta: len(sequence)]

    def pad_sequences(self, sequences: list, maxlen: int, padding: str = 'post', truncating: str = 'post') -> np.ndarray:
        result = []
        for _, sequence in enumerate(sequences):
            delta = sequence.shape[0] - maxlen
            if delta < 0:
                sequence = self.padding_sequence(sequence, padding, maxlen)
            elif delta > 0:
                sequence = self.truncating_sequence(sequence, truncating, maxlen)
            result.append(sequence)
        
        return np.array(result)