.PHONY: train
train:
	rm -rf ./output
	# allennlp train -s=./output ./configs/bilstm_tagger.json
	python train_classifier.py > log.txt