# Shopee Product Matching — Track 1 Kaggle Benchmark Deep Dive

## Competition
Shopee - Price Match Guarantee  
https://www.kaggle.com/competitions/shopee-product-matching

## Dataset
Dataset download here
https://www.kaggle.com/competitions/shopee-product-matching/data
then either its stored inside kaggle folder or save it locally inside this folder

## Problem
Given ~34,250 product listings each with an image + text title,
identify which listings belong to the same product group.
Metric: Mean F1 score.

## Experiments
- Exp A: Image encoder ablation (EfficientNet B0 / B3 / B5)
- Exp B: Text encoder ablation (TF-IDF / MiniLM / XLM-R)
- Exp C: ArcFace margin ablation (m = 0.3 / 0.5 / 0.7)

## Run First Time
python --version
python -m venv venv
pip install -r requirements.txt

## Run
python src/utils.py

## Run jupyter notebook
venv\Scripts\activate
jupyter notebook

## Results
See results/ folder
