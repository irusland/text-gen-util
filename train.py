import argparse
import logging
import os
from pathlib import Path

from src.dictionary import Dictionary
from src.model import NGramModel
from src.trainer import Trainer

FORMAT = '%(message)s'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_dir(path):
    if not os.path.isdir(path):
        raise argparse.ArgumentError(None, f'{path} is not directory')
    return path


parser = argparse.ArgumentParser(description='Utility for text generation model training.')
parser.add_argument('--input-dir',
                    type=is_dir,
                    required=True,
                    help='Directory for training texts. If not set, then stdin is used.')
parser.add_argument('--model', type=Path,
                    default='model.pkl',
                    help='File for saving model.')

args = parser.parse_args()
model_path = args.model

ngram_model = NGramModel()
dictionary = Dictionary()
trainer = Trainer(ngram_model=ngram_model, dictionary=dictionary)

trainer.fit(input_dir=args.input_dir)

trainer.save(model_path)

