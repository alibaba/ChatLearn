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
		$(shell git ls-tree --full-tree --name-only -r HEAD rlhf | grep \.py$) \
		$(shell git diff --cached --name-only rlhf | grep \.py$)


.DEFAULT_GOAL := lint
