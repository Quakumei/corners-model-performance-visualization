DOWNLOAD_LINK = "https://ai-process-sandy.s3.eu-west-1.amazonaws.com/purge/deviation.json"

.PHONY: all help install_deps lint black isort flake8 build_plots

all: help

help:
	@echo "Available targets:"
	@echo "  install_deps - Install dependencies"
	@echo "  lint         - Run code linting"
	@echo "  build_plots  - Build plots using data/deviation.json"
	@echo "  clean        - Clean generated files"

data/deviation.json:
	curl --output data/deviation.json --url $(DOWNLOAD_LINK)

build_plots: data/deviation.json
	python3 -m poetry run src/CornersModelPerformancePlotter.py --data-filename data/deviation.json --output-dir plots --darkgrid

install_deps:
	python3 -m poetry install

black:
	python3 -m poetry run black . --line-length=79

isort:
	python3 -m poetry run isort .

flake8:
	python3 -m poetry run flake8 .

lint: black isort flake8
	@echo "Linting done"

clean:
	rm -f data/deviation.json
	rm -f plots/*.png
