clean: 
	pip uninstall -y deepplantphenomics

documentation:
	mkdocs build

install:
	pip install .

test:
	python -m pytest .
