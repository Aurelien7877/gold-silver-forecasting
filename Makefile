PYTHON := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: install install-research test lint download analysis train predict figures export-hf

install:
	$(PIP) install -e '.[dev]'

install-research:
	$(PIP) install -e '.[dev,research]'

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check src tests scripts

download:
	PYTHONPATH=src $(PYTHON) scripts/download_data.py

analysis:
	PYTHONPATH=src $(PYTHON) scripts/run_analysis.py

train:
	PYTHONPATH=src $(PYTHON) scripts/train.py

predict:
	PYTHONPATH=src $(PYTHON) scripts/predict.py

figures:
	MPLBACKEND=Agg PYTHONPATH=src $(PYTHON) scripts/generate_report_figures.py

export-hf:
	PYTHONPATH=src $(PYTHON) scripts/export_hf.py
