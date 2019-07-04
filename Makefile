init:
	pip3 install -r requirements.txt

test:
	python3 -m unittest tests/simple_unittest.py

train:
	python3 -m unittest tests/train_unittest.py

evaluate:
	python3 -m unittest tests/evaluate_unittest.py

predict:
	python3 -m unittest tests/predict_unittest.py

lint: 
	pylint --rcfile=.pylintrc main.py

run:
	python3 main.py

.PHONY: init test run
