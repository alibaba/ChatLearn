PYTHON ?= python
ADDITIONAL_DEPS ?=

.PHONY: build
build: $(LIB)
	$(PYTHON) setup.py bdist_wheel --universal
	@printf "\033[0;32mPIP package built\033[0m: "
	@ls dist/*.whl

.PHONY: test
test: $(LIB)
	cd tests; bash run_tests.sh

.PHONY: lint
lint:
	pip install pylint==2.16.1
	@$(PYTHON) -m pylint \
		--rcfile=.pylintrc --output-format=parseable --jobs=8 \
		$(shell git ls-tree --full-tree --name-only -r HEAD chatlearn | grep \.py$) \
		$(shell git diff --cached --name-only chatlearn | grep \.py$) \
		$(shell git ls-tree --full-tree --name-only -r HEAD examples/megatron/ | grep \.py$) \
		$(shell git diff --cached --name-only examples/megatron/ | grep \.py$) \

.PHONY: doc
doc:
	cd docs; make html


.DEFAULT_GOAL := lint
