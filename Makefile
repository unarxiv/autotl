default:
	@echo "Usage:"
	@echo "\tmake test"
	@echo "\tmake format"
	@echo "\tmake docs"

test:


format:
	autoflake -i autotl/*.py
	autoflake -i autotl/**/*.py

	isort -i autotl/*.py
	isort -i autotl/**/*.py 

	yapf -i autotl/*.py
	yapf -i autotl/**/*.py

docs:
	cd docs && npm run docs:build

.PHONY: docs