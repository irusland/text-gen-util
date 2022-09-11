import itertools
import re
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset

from src.dictionary import Dictionary


class TokenDataset(Dataset):
    def __init__(
        self,
        dictionary: Dictionary,
        raw_text: str,
        ngram: int = 2,
        min_word_length: int = 1,
    ):
        self._dictionary = dictionary
        self._whitespace = re.compile(r"\s")
        self._digits = re.compile(r"\d")
        self._latin = re.compile(r"[a-z]", flags=re.IGNORECASE)
        self._sentence_terminators = re.compile(r"[^\s\w]|[_]")

        self._clean_text = self._preprocess(raw_text)
        self._ngram = ngram
        self._min_word_length = min_word_length

        self._index_to_sentence_map, self._sentences = self._split_into_sentences(self._clean_text)

        self._sentences_into_dictionary(self._sentences)

    def get_tokens(self) -> List[int]:
        return [
            self._dictionary.encode(word) for word in itertools.chain(*self._sentences)
        ]

    def _preprocess(self, raw_text: str) -> str:
        text = raw_text.lower()
        text = self._digits.sub(' ', text)
        text = self._latin.sub(' ', text)
        return text

    def _make_ngrams(self, words: Tuple[str], initial_index: int, index_to_sentence_map) -> int:
        processed = 0
        for ngram_len in range(1, self._ngram + 1):
            ngram_count = len(words) - ngram_len
            for index in range(ngram_count):
                index_to_sentence_map[initial_index + index] = (
                    words[index:index + ngram_len],
                    words[index + ngram_len:index + ngram_len + 1],
                )
            initial_index += ngram_count
            processed += ngram_count
        return processed

    def _split_into_sentences(self, text: str) -> Tuple[Dict[int, Tuple[Tuple[str], Tuple[str]]], List[List[str]]]:
        sentences = self._sentence_terminators.split(text)
        resulting_sentences = []
        pair_count = 0
        index_to_sentence_map = {}
        for sentence in sentences:
            if not (words := self._split_into_words(sentence)):
                continue
            if len(words) <= self._ngram:
                resulting_sentences.append(words)
                continue
            processed = self._make_ngrams(words, pair_count, index_to_sentence_map)
            pair_count += processed
            resulting_sentences.append(words)

        return index_to_sentence_map, resulting_sentences

    def _split_into_words(self, sentence: str) -> Tuple[str]:
        words = self._whitespace.split(sentence)
        # todo lemmatize?

        def filter_(word):
            return len(word) > self._min_word_length and word != ''
        return tuple(filter(filter_, words))

    def _sentences_into_dictionary(self, sentences):
        self._dictionary.transform(itertools.chain(*sentences))

    def __len__(self):
        return len(self._index_to_sentence_map)

    def __getitem__(self, index):
        X, Y = self._index_to_sentence_map[index]
        return (
            torch.tensor(
                [self._dictionary.encode(x) for x in X]
            ),
            torch.tensor(
                [self._dictionary.encode(y) for y in Y]
            ),
        )
