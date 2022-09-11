import fileinput
import logging
import pickle
from pathlib import Path
from typing import Optional, List

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import TokenDataset
from src.dictionary import Dictionary
from src.model import NGramModel, NGramModelError

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, ngram_model: NGramModel, dictionary: Dictionary, ngram: int = 2):
        self._ngram_model = ngram_model
        self._ngram = ngram
        self._dictionary = dictionary

    def fit(self, input_dir: Optional[str]):
        texts = []
        if input_dir is not None:
            input_dir = Path(input_dir)
            queue = list(input_dir.glob(pattern='*'))
            for path in queue:
                if not path.is_file():
                    queue.extend(path.glob(pattern='*'))
                    continue
                raw_text = path.read_text()
                texts.append(raw_text)
        else:
            for line in fileinput.input():
                raw_text = line
                texts.append(raw_text)
                break
        self._run_iterative(texts=texts)

    def _run_iterative(self, texts: List[str]):
        for text in tqdm(texts):
            dataloader = self._prepare_dataloader(raw_text=text)
            self._fit_iteration(dataloader)

    def _prepare_dataloader(self, raw_text: str) -> DataLoader:
        dataset = TokenDataset(raw_text=raw_text, ngram=self._ngram, dictionary=self._dictionary)
        return DataLoader(dataset, batch_size=1)

    def _fit_iteration(self, dataloader: DataLoader) -> None:
        self._ngram_model.fit(dataloader)

    def _pretty_sentence(self, sentence_tokens: List[int]) -> str:
        sentence = tuple(self._dictionary.decode(token) for token in sentence_tokens)
        return f"{' '.join(sentence).capitalize()}."

    def _pretty_text(self, sentences: List[List[int]]) -> str:
        return '\n'.join([self._pretty_sentence(sentence) for sentence in sentences])

    def _get_sentence_tokens(self, sentence: Optional[List[str]]):
        if sentence is None:
            sentence_tokens = self._ngram_model.random_ngram()
            logger.info(
                'Random sentence for generation: %s',
                ' '.join(self._dictionary.decode_many(sentence_tokens))
            )
        else:
            logger.info('Input sentence for generation: %s', ' '.join(sentence))
            sentence_tokens = TokenDataset(raw_text=' '.join(sentence), dictionary=self._dictionary).get_tokens()
        return sentence_tokens

    def continue_(self, sentence: Optional[List[str]], word_count: int):
        sentence_tokens = self._get_sentence_tokens(sentence)
        logger.info('Target length: %s', word_count)
        logger.debug('Dictionary len = %s', len(self._dictionary))

        result_text = []
        current_sentence = []
        current_sentence.extend(sentence_tokens[:word_count])
        if len(sentence_tokens) >= word_count:
            return current_sentence
        word_to_continue_left = word_count - len(sentence_tokens)
        logger.debug('Starting with %s', self._dictionary.decode_many(current_sentence))
        logger.debug('Words to generate %s', word_to_continue_left)
        for i in range(word_to_continue_left):
            tokens_to_continue = tuple(current_sentence[-self._ngram:])
            logger.debug('Generating token %s for %s (%s)', i+1, tokens_to_continue, self._dictionary.decode_many(tokens_to_continue))
            try:
                next_token = self._ngram_model.samples(tokens_to_continue, k=1)
                current_sentence.extend(next_token)
            except NGramModelError:
                next_token = self._ngram_model.random_ngram()
                logger.debug('Cannot continue, starting new sentence with %s', self._dictionary.decode_many(next_token))
                result_text.append(current_sentence)
                current_sentence = list(next_token)
        result_text.append(current_sentence)
        return self._pretty_text(result_text)

    def save(self, path: Path) -> None:
        to_dump = self
        with path.open('wb') as fp:
            pickle.dump(to_dump, fp)

    @classmethod
    def load(cls, path: Path) -> 'Trainer':
        with path.open('rb') as fp:
            dumped = pickle.load(fp)

        return dumped
