import argparse
import logging
import os
from pathlib import Path

from src.trainer import Trainer


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
                    default=7,
                    help='Word of resulting sentence.')
parser.add_argument('--log-level',
                    default='ERROR',
                    choices=logging._nameToLevel.keys(),
                    help='Print debug messages')
args = parser.parse_args()
FORMAT = '%(message)s'
level = logging._nameToLevel[args.log_level]

logging.basicConfig(level=level)
logger = logging.getLogger(__name__)

model_path = args.model
sentence_prefix = args.prefix
word_count = args.length

trainer = Trainer.load(model_path)

result = trainer.continue_(sentence=sentence_prefix, word_count=word_count)

logger.debug("Resulting sentence: %s", result)
print(result)
