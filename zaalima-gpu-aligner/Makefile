PY ?= python3
.PHONY: install build web test demo bench info clean

install:            ## install deps + build Cython extension
	$(PY) -m pip install -r requirements.txt
	$(MAKE) build

build:              ## compile the Cython FM-index core (optional, ~20x faster seeding)
	$(PY) setup.py build_ext --inplace

test:               ## full test suite (kernel equivalence also runs on CUDA simulator)
	PYTHONPATH=. $(PY) -m pytest -q

web:                ## start the web application on http://localhost:8000
	$(PY) run_web.py

info:               ## show backends / GPU / build status
	PYTHONPATH=. $(PY) -m gpualign info

demo:               ## simulate data -> index -> align -> VCF with accuracy report
	bash scripts/demo.sh

bench:              ## kernel + end-to-end benchmark report
	PYTHONPATH=. $(PY) -m gpualign benchmark --out bench_out

clean:
	rm -rf build dist *.egg-info demo_out bench_out .pytest_cache gpualign/index/_fmcore.c gpualign/index/*.so
	find . -name __pycache__ -type d -exec rm -rf {} +
