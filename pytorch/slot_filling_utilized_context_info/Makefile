.PHONY: one
one:
	conda activate allennlp-0.9
	allennlp train -s=./output/bilstm_fc -r ./bilstm_fc.json

.PHONY: two
two:
	allennlp train -s=./output/bilstm_crf -r --include-package=./packages ./bilstm_crf.json