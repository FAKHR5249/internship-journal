# ===================================
# Advanced Regression Project
# ===================================

load:
	python -m src.data.load_data

validate:
	python -m src.data.validate_data

features:
	python -m src.features.build_features

train:
	python -m src.models.train

tune:
	python -m src.models.tune

evaluate:
	python -m src.models.evaluate

predict:
	python -m src.models.predict

plots:
	python -m src.visualization.plots

all:
	python -m src.data.load_data
	python -m src.data.validate_data
	python -m src.features.build_features
	python -m src.models.train
	python -m src.models.tune
	python -m src.models.evaluate
	python -m src.models.predict
	python -m src.visualization.plots