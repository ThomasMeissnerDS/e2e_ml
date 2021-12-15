.PHONY: docs
docs:
	poetry run sphinx-apidoc -o source ../e2eml
	poetry run sphinx-build -M docs source build -b html
