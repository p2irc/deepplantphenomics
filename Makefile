clean: 
	find . -name '*.pyc' -delete
	rm -r ./site

documentation:
	mkdocs build