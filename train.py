import argparse
import logging
import os
from pathlib import Path

from pyfillet import WordEmbedder

from src.dictionary import Dictionary
from src.model import NGramModel
from src.trainer import Trainer

FORMAT = '%(message)s'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(description='Utility for text generation model training.')
parser.add_argument('--input-dir',
                    default=None,
                    help='Directory for training texts. If not set, then stdin is used.')
parser.add_argument('--model', type=Path,
                    default='model.pkl',
                    help='File for saving model.')
parser.add_argument('--ngram', type=int,
                    default=2,
                    help='Ngram count.')
parser.add_argument('--min-ngram', type=int,
                    default=1,
                    help='Minimum ngram length.')
parser.add_argument('--nsamples', type=int,
                    default=3,
                    help='Max next word samples to choose from.')


def main():
    args = parser.parse_args()
    model_path = args.model

    ngram_model = NGramModel()
    dictionary = Dictionary()
    embedder = WordEmbedder()
    trainer = Trainer(
        ngram_model=ngram_model,
        dictionary=dictionary,
        embedder=embedder,
        ngram=args.ngram,
        min_ngram=args.min_ngram,
        nsamples=args.nsamples,
    )

    trainer.fit(input_dir=args.input_dir)

    trainer.save(model_path)


if __name__ == '__main__':
    main()
