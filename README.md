# Shopee Product Matching — Track 1 Kaggle Benchmark Deep Dive

## Competition
Shopee - Price Match Guarantee  
https://www.kaggle.com/competitions/shopee-product-matching

## Problem
Given ~34,250 product listings each with an image + text title,
identify which listings belong to the same product group.
Metric: Mean F1 score.

## Baseline
Silver Medal solution by mfalfafa:
https://github.com/mfalfafa/shopee-price-match-guarantee

## Experiments
- Exp A: Image encoder ablation (EfficientNet B0 / B3 / B5)
- Exp B: Text encoder ablation (TF-IDF / MiniLM / XLM-R)
- Exp C: ArcFace margin ablation (m = 0.3 / 0.5 / 0.7)

## How to Run
pip install -r requirements.txt
python src/evaluate.py

## Results
See results/ablation_results.csv
