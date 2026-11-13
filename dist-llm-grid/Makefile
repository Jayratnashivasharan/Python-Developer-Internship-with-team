.PHONY: install proto model test run load docker-up lint clean
PY ?= python

install:        ## install dependencies
	$(PY) -m pip install -r requirements-dev.txt
proto:          ## regenerate gRPC stubs after editing grid/proto/inference.proto
	$(PY) -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. grid/proto/inference.proto
model:          ## create the tiny random demo checkpoint
	$(PY) -m grid.weights.make_tiny_model --out models/tiny-llama
test:           ## run the full test-suite
	$(PY) -m pytest tests -q
run:            ## start 2 workers + gateway locally (http://localhost:8000)
	$(PY) run.py
load:           ## load-test the running cluster
	$(PY) scripts/load_test.py --concurrency 32 --requests 128 --out docs/load_test_latest.json
docker-up:      ## docker compose cluster
	docker compose -f deploy/docker/docker-compose.yml up --build
lint:
	ruff check grid tests scripts
clean:
	rm -rf models __pycache__ .pytest_cache
