
.PHONY: docs
docs: 
	cd docs/ && \
	python make_docs.py --sphinx --parallel

.PHONY: paper
paper:
	cd paper/ && \
	./joss_make_latex.sh
