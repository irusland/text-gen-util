import os
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest



TRAIN_TEXT = """
Привет всем меня звать Руслан! Хочешь знать чтобы всё знать?
Привет всем меня звать Руслан Хочешь знать чтобы всё знать Тогда тебе вообще-то -  сюда Привет всем меня звать Руслан Хочешь знать Привет всем меня звать Руслан Хочешь знать Привет всем меня звать Руслан Хочешь знать

Тогда тебе вообще-то -  сюда


раз два три четыре
раз два четыре123 ojaposd opa w

askp[d}{AS}D{
"""


@pytest.fixture(params=["test_data"])
def temp_dir(tmpdir, request):
    d = tmpdir.mkdir(request.param)
    f1 = d.join("myfile")
    f1.write(TRAIN_TEXT)
    assert f1.read() == TRAIN_TEXT
    return d


@pytest.fixture()
def train_arguments(request):
    args = ['train.py', *request.param.split()]
    with patch.object(sys, 'argv', args):
        yield


@pytest.fixture()
def stdin(request):

    with patch.object(sys, 'stdin', StringIO(request.param)):
        yield


@pytest.fixture(params=['./model.pkl'])
def assert_model_path(request):
    yield
    filepath = Path(request.param)
    assert filepath.exists()
    os.remove(filepath)
