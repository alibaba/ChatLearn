PYTHON ?= python3
ADDITIONAL_DEPS ?=
current_dir := $(shell pwd | sed 's:/*$$::')

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
	git config --global --add safe.directory $(current_dir)
	@$(PYTHON) -m pip install pylint==2.16.1  -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
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
