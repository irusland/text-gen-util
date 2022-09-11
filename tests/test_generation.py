import sys
from unittest.mock import patch

import pytest

from train import main as train_main
from generate import main


@pytest.fixture()
def prepare(stdin, train_arguments):
    train_main()


@pytest.fixture()
def generate_arguments(request, prepare, capsys):
    out, err = capsys.readouterr()
    args = ['generate.py', *request.param.split()]
    with patch.object(sys, 'argv', args):
        yield


class TestGeneration:
    @pytest.mark.parametrize(
        'train_arguments',
        ['--ngram 4 --min-ngram=1'],
        indirect=True
    )
    @pytest.mark.parametrize(
        'generate_arguments',
        ['--prefix один --length 100'],
        indirect=True,
    )
    @pytest.mark.parametrize(
        'stdin',
        [
         """один два три два
два три три три три три три три три четыре"""
         ],
        indirect=True
    )
    def test_generate(self, prepare, stdin, assert_model_path, capsys, generate_arguments, train_arguments):
        main()

        out, err = capsys.readouterr()
        assert err == ''
        assert out
        assert out.startswith('Один')
        assert out.endswith('.\n')
        assert len(out.split()) == 100
