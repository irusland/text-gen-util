import fileinput
import logging
import pickle
import sys
from math import inf
from pathlib import Path
from typing import Optional, List

import numpy as np
from numpy import ndarray
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import TokenDataset
from src.dictionary import Dictionary
from src.model import NGramModel, NGramModelError

from pyfillet import WordEmbedder

from src.utils import angle_between

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        ngram_model: NGramModel,
        dictionary: Dictionary,
        embedder: WordEmbedder,
        ngram: int = 2,
        min_ngram: int = 1,
        nsamples: int = 10,
    ):
        self._ngram_model = ngram_model
        self._dictionary = dictionary
        self._embedder = embedder
        self._ngram = ngram
        self._min_ngram = min_ngram
        self._nsamples = nsamples

    def fit(self, input_dir: Optional[str]):
        texts = []
        if input_dir is not None:
            input_dir = Path(input_dir)
            queue = list(input_dir.glob(pattern="*"))
            for path in queue:
                if not path.is_file():
                    queue.extend(path.glob(pattern="*"))
                    continue
                raw_text = path.read_text()
                texts.append(raw_text)
        else:
            for line in sys.stdin:
                raw_text = line
                texts.append(raw_text)
                break
        self._run_iterative(texts=texts)

    def _run_iterative(self, texts: List[str]):
        for text in tqdm(texts):
            dataloader = self._prepare_dataloader(raw_text=text)
            self._fit_iteration(dataloader)

    def _prepare_dataloader(self, raw_text: str) -> DataLoader:
        dataset = TokenDataset(
            raw_text=raw_text,
            dictionary=self._dictionary,
            ngram=self._ngram,
            min_ngram=self._min_ngram,
        )
        return DataLoader(dataset, batch_size=1)

    def _fit_iteration(self, dataloader: DataLoader) -> None:
        self._ngram_model.fit(dataloader)

    def _pretty_sentence(self, sentence_tokens: List[int]) -> str:
        sentence = tuple(self._dictionary.decode(token) for token in sentence_tokens)
        return f"{' '.join(sentence).capitalize()}."

    def _pretty_text(self, sentences: List[List[int]]) -> str:
        return "\n".join([self._pretty_sentence(sentence) for sentence in sentences])

    def _get_sentence_tokens(self, sentence: Optional[List[str]]):
        if sentence is None:
            sentence_tokens = self._ngram_model.random_ngram()
            logger.debug(
                "Random sentence for generation: %s",
                " ".join(self._dictionary.decode_many(sentence_tokens)),
            )
        else:
            logger.debug("Input sentence for generation: %s", " ".join(sentence))
            sentence_tokens = TokenDataset(
                raw_text=" ".join(sentence), dictionary=self._dictionary
            ).get_tokens()
        return sentence_tokens

    def _embed_tokens(self, tokens: List[int]) -> List[ndarray]:
        words = self._dictionary.decode_many(tokens)
        return [self._embedder(word=word) for word in words]

    def _get_sentence_embedding(self, sentence: List[int]) -> ndarray:
        embeddings = self._embed_tokens(tokens=sentence)
        return np.sum(embeddings, axis=0)

    def _safe_sum(self, a: ndarray, b: Optional[ndarray]) -> ndarray:
        if b is None:
            return a
        return a + b

    def _choose_closest_next_token(
        self, next_tokens: List[int], current_theme_vector: ndarray
    ) -> (int, List[float]):
        embeddings = self._embed_tokens(next_tokens)

        def get_angle(iv):
            _, v = iv
            if v is None:
                return np.inf
            angle = angle_between(current_theme_vector, v)
            return angle

        i, closest = min(
            enumerate(embeddings),
            key=get_angle,
        )
        next_words = self._dictionary.decode_many(next_tokens)
        logger.debug(
            "Closest token to current theme out of %s %s is %s",
            len(embeddings),
            next_words,
            next_words[i],
        )
        current_theme_vector = self._safe_sum(current_theme_vector, closest)
        return next_tokens[i], current_theme_vector

    def _generate_text(
        self, word_to_continue_left: int, base_sentence: List[int]
    ) -> List[List[int]]:
        result_text = []
        current_sentence = base_sentence
        current_theme_vector = np.zeros(
            self._embedder.dim,
        )
        current_theme_vector = self._safe_sum(
            current_theme_vector, self._get_sentence_embedding(current_sentence)
        )
        for i in range(word_to_continue_left):
            tokens_to_continue = tuple(current_sentence[-self._ngram :])
            logger.debug(
                "Generating token %s for %s (%s)",
                i + 1,
                tokens_to_continue,
                self._dictionary.decode_many(tokens_to_continue),
            )
            try:
                next_tokens = self._ngram_model.samples(
                    tokens_to_continue, k=self._nsamples
                )
                next_token, current_theme_vector = self._choose_closest_next_token(
                    next_tokens=next_tokens,
                    current_theme_vector=current_theme_vector,
                )
                current_sentence.append(next_token)
            except NGramModelError:
                next_tokens = self._ngram_model.random_ngram()
                logger.debug(
                    "Cannot continue, starting new sentence with %s",
                    self._dictionary.decode_many(next_tokens),
                )
                result_text.append(current_sentence)
                current_sentence = list(next_tokens)
        result_text.append(current_sentence)
        return result_text

    def continue_(self, sentence: Optional[List[str]], word_count: int):
        sentence_tokens = self._get_sentence_tokens(sentence)
        logger.debug("Target length: %s", word_count)
        logger.debug("Dictionary len = %s", len(self._dictionary))

        current_sentence = []
        current_sentence.extend(sentence_tokens[:word_count])
        word_to_continue_left = word_count - len(sentence_tokens)
        logger.debug("Starting with %s", self._dictionary.decode_many(current_sentence))
        logger.debug("Words to generate %s", word_to_continue_left)
        result_text = self._generate_text(
            word_to_continue_left=word_to_continue_left, base_sentence=current_sentence
        )
        return self._pretty_text(result_text)

    def save(self, path: Path) -> None:
        to_dump = self
        with path.open("wb") as fp:
            pickle.dump(to_dump, fp)

    @classmethod
    def load(cls, path: Path) -> "Trainer":
        with path.open("rb") as fp:
            dumped = pickle.load(fp)

        return dumped
