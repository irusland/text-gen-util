format:
	python -m black . --exclude=tests

example:
	echo 'привет всем всем привет' | python train.py --ngram 4 --min-ngram=1 --model ./sample.pkl
	python generate.py --prefix 'привет' --length 10 --model ./sample.pkl
