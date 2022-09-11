# Утилита для генерации предложений

## setup
```bash
poetry install
```

## example
```bash
make example
```
runs
```bash
echo 'привет всем всем привет' | python train.py --ngram 4 --min-ngram=1 --model ./sample.pkl
python generate.py --prefix 'привет' --length 10 --model ./sample.pkl
```
outputs
```
Привет всем всем привет всем всем привет всем всем привет.
```

## Training utility train.py
```
Utility for text generation model training.

optional arguments:
  -h, --help            show this help message and exit
  --input-dir INPUT_DIR
                        Directory for training texts. If not set, then stdin is used.
  --model MODEL         File for saving model.
  --ngram NGRAM         Ngram count.
  --min-ngram MIN_NGRAM
                        Minimum ngram length.
  --nsamples NSAMPLES   Max next word samples to choose from.
```

## Generation utility generate.py
```
Utility for text generation model evaluation.

optional arguments:
  -h, --help            show this help message and exit
  --prefix PREFIX [PREFIX ...]
                        First words of sentence.
  --model MODEL         File for loading model.
  --length LENGTH       Word of resulting sentence.
  --log-level {CRITICAL,FATAL,ERROR,WARN,WARNING,INFO,DEBUG,NOTSET}
                        Print debug messages
```

## Train data gathering crawl.py
Loads wikipedia pages
