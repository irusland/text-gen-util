import pytest

from tests.conftest import TRAIN_TEXT
from train import main


class TestTrain:
    @pytest.mark.parametrize(
        'train_arguments',
        ['--input-dir test_data --ngram 4 --min-ngram=1',],
        indirect=True
    )
    def test_train(self, train_arguments, assert_model_path):
        main()

    @pytest.mark.parametrize(
        'train_arguments',
        ['--ngram 4 --min-ngram=1',],
        indirect=True
    )
    @pytest.mark.parametrize(
        'stdin',
        ['', TRAIN_TEXT,
         """1 строка один сейчас
строка два потом
строка три потом
"""],
        indirect=True
    )
    def test_train_stdin(self, train_arguments, stdin, assert_model_path):
        main()
