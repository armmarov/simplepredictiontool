init:
	pip install -r requirements.txt

test:
	python3 -m unittest tests/simple_unittest.py

.PHONY: init test
