default:
	@echo "Usage:"
	@echo "\tmake test
	@echo "\tmake format

test:


format:
	autoflake -i autotl/*.py
	autoflake -i autotl/**/*.py

	isort -i autotl/*.py
	isort -i autotl/**/*.py 

	yapf -i autotl/*.py
	yapf -i autotl/**/*.py
