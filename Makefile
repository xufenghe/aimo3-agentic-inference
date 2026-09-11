.PHONY: check demo test lint build clean

check: lint test build demo

lint:
	ruff check .

test:
	python -m unittest discover -s tests -v

build:
	python -m build

demo:
	aimo3 demo

clean:
	python -c "import shutil; [shutil.rmtree(path, ignore_errors=True) for path in ('build', 'dist')]"
