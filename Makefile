VENV ?= $(HOME)/venvs/sexvary
PYTHON ?= $(VENV)/bin/python
REMOTE_HOST ?=

.PHONY: install validate test demo checklist local-nlsy piaac pisa timss pirls icils nhanes nnyfs psid hrs nces-school compare-results paper-report backend backend-health backend-restore backend-sync-health update-site-assets

install:
	uv pip install --python "$(PYTHON)" -e .[dev]

validate:
	$(PYTHON) scripts/validate_project.py

test:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/sexvary-pycache PYTEST_ADDOPTS='-p no:cacheprovider' $(PYTHON) -B -m pytest tests/ -q

demo:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/sexvary-pycache $(PYTHON) scripts/run_demo_pipeline.py

checklist:
	$(PYTHON) scripts/acquire_public_data.py --write-checklist

local-nlsy:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/sexvary-pycache $(PYTHON) scripts/run_local_nlsy_pipeline.py

piaac:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/sexvary-pycache $(PYTHON) scripts/run_piaac_pipeline.py

pisa:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/sexvary-pycache $(PYTHON) scripts/run_pisa_pipeline.py

timss:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/sexvary-pycache $(PYTHON) scripts/run_timss_pipeline.py --dataset-id $(or $(DATASET_ID),timss_2019)

pirls:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/sexvary-pycache $(PYTHON) scripts/run_pirls_pipeline.py

icils:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/sexvary-pycache $(PYTHON) scripts/run_icils_pipeline.py

nhanes:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/sexvary-pycache $(PYTHON) scripts/run_nhanes_pipeline.py

nnyfs:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/sexvary-pycache $(PYTHON) scripts/run_nnyfs_pipeline.py

psid:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/sexvary-pycache $(PYTHON) scripts/run_psid_pipeline.py

hrs:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/sexvary-pycache $(PYTHON) scripts/run_hrs_pipeline.py

nces-school:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/sexvary-pycache $(PYTHON) scripts/run_nces_school_pipeline.py --dataset-id $(DATASET_ID)

compare-results:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/sexvary-pycache $(PYTHON) scripts/run_cross_dataset_comparison.py

paper-report:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/sexvary-pycache $(PYTHON) scripts/run_paper_bundle.py

backend:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/sexvary-pycache $(PYTHON) scripts/run_backend.py

backend-health:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/sexvary-pycache $(PYTHON) scripts/run_backend_health_report.py --remote-host $(REMOTE_HOST)

backend-restore:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/sexvary-pycache $(PYTHON) scripts/run_backend_restore.py $(if $(PIPELINE_ID),--pipeline-id $(PIPELINE_ID),)

backend-sync-health:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/sexvary-pycache $(PYTHON) scripts/run_backend_sync_health.py --remote-host $(REMOTE_HOST)

update-site-assets:
	cp results/figures/forest_log_variance_ratio_primary.png site/assets/img/
	cp results/figures/forest_log_variance_ratio_secondary.png site/assets/img/
	cp results/figures/dataset_family_summary.png site/assets/img/
	cp results/figures/age_profile_log_variance_ratio.png site/assets/img/
	cp results/figures/mean_vs_variance_scatter.png site/assets/img/
	cp results/figures/robustness_comparison.png site/assets/img/
