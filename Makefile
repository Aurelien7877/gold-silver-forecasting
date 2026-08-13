PYTHON := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: install install-research test lint download analysis train foundation-benchmark joint-benchmark global-benchmark robustness predict figures export-hf

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

foundation-benchmark:
	PYTHONPATH=src $(PYTHON) scripts/benchmark_foundation_models.py \
		--chronos-path "$${GOLD_SILVER_CHRONOS_PATH}" \
		--timesfm-path "$${GOLD_SILVER_TIMESFM_PATH}"

joint-benchmark:
	PYTHONPATH=src $(PYTHON) scripts/benchmark_joint_model.py

global-benchmark:
	PYTHONPATH=src $(PYTHON) scripts/benchmark_global_model.py

robustness:
	PYTHONPATH=src $(PYTHON) scripts/run_robustness.py

predict:
	PYTHONPATH=src $(PYTHON) scripts/predict.py

figures:
	MPLBACKEND=Agg PYTHONPATH=src $(PYTHON) scripts/generate_report_figures.py

export-hf:
	PYTHONPATH=src $(PYTHON) scripts/export_hf.py
