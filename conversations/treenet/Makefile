help:
	@echo "    train-core"
	@echo "        Train a dialogue model using Rasa core."

train:
	python -m rasa_core.train -s data/core/ -d domain.yml -o models/dialogue -c policy_config.yml --augmentation 0
