import argparse
import logging
import os
from pathlib import Path

from src.model import NGramModel
from src.trainer import Trainer

FORMAT = '%(message)s'
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def is_dir(path):
    if not os.path.isdir(path):
        raise argparse.ArgumentError(None, f'{path} is not directory')
    return path


parser = argparse.ArgumentParser(description='Utility for text generation model evaluation.')
parser.add_argument('--prefix',
                    type=str,
                    nargs='+',
                    help='First words of sentence.')
parser.add_argument('--model', type=Path,
                    default='model.pkl',
                    help='File for loading model.')
parser.add_argument('--length', type=int,
                    default=5,
                    help='Word of resulting sentence.')

args = parser.parse_args()
# print(args)
model_path = args.model
sentence_prefix = args.prefix
word_count = args.length

trainer = Trainer.load(model_path)

result = trainer.continue_(sentence=sentence_prefix, word_count=word_count)

logger.info("Resulting sentence: %s", result)
