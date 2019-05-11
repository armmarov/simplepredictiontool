init:
	pip3 install -r requirements.txt

test:
	python3 -m unittest tests/simple_unittest.py

run:
	python3 main.py

.PHONY: init test run
