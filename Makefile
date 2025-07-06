.PHONY: start stop build test clean

start:
	docker-compose up -d

stop:
	docker-compose down

build:
	docker-compose build

test:
	python test.py

clean:
	docker-compose down -v
