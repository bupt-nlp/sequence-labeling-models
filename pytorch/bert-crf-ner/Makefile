.PHONY: run0
run0:
	rm -rf ./output
	allennlp train -s=./output --include-package=allen ./config/event_trigger_classifier.json 

.PHONY: clean
clean:
	rm -rf ./output

.PHONY: simple
simple:
	rm -rf ./output/simple_tagger
	allennlp train -s=./output/simple_tagger --include-package=allen ./config/simple_tagger.json
