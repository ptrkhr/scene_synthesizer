
.PHONY: docs
docs: 
	cd docs/ && \
	python make_docs.py --sphinx --parallel

.PHONY: paper
paper:
	cd paper/ && \
	./joss_make_latex.sh

.PHONY: paperdraft
paperdraft:
	docker run --rm --volume $(PWD)/paper:/data --user $(id -u):$(id -g) --env JOURNAL=joss openjournals/inara
