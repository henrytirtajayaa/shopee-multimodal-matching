import os
import cv2
import math
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.utils.data import Dataset
from torchvision import transforms
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors


# CONFIG ===

class CFG:

    DATA_DIR = '/kaggle/input/shopee-product-matching' \
           if os.path.exists('/kaggle') else '../shopee-product-matching'
    
    TRAIN_CSV       = os.path.join(DATA_DIR, 'train.csv')
    TRAIN_IMG_DIR   = os.path.join(DATA_DIR, 'train_images')
    RESULTS_DIR     = '../results'

    # Model defaults (overridden per experiment)
    IMG_MODEL       = 'efficientnet_b5'   # swap in ablation A
    IMG_SIZE        = 512
    EMBED_DIM       = 512
    BATCH_SIZE      = 16
    EPOCHS          = 5
    LR              = 1e-4
    DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ArcFace defaults (overridden in ablation C)
    ARC_S           = 30.0
    ARC_M           = 0.5

    # TEXT_MODEL      = 'paraphrase-xlm-r-multilingual-v1'
    TEXT_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'


# DATASET ===

class ShopeeDataset(Dataset):
    """
    Loads (image, label) pairs for training.
    label_group strings are mapped to integer indices.
    """
    def __init__(self, df, img_dir, transform=None):
        self.df        = df.reset_index(drop=True)
        self.img_dir   = img_dir
        self.transform = transform

        # Map label_group strings → integer class indices
        unique_groups  = df['label_group'].unique()
        self.label_map = {g: i for i, g in enumerate(unique_groups)}
        self.num_classes = len(unique_groups)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = self.label_map[row['label_group']]

        # Load image
        img_path = os.path.join(self.img_dir, row['image'])
        image    = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def get_transforms(img_size=512, mode='train'):
    """Standard augmentation pipeline."""
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])


# ARCFACE LOSS ===

class ArcFaceLoss(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss for metric learning.
    Pushes same-class embeddings closer on a hypersphere.
    Paper: https://arxiv.org/abs/1801.07698

    Args:
        in_features:  embedding dimension
        num_classes:  number of unique label groups
        s:            feature scale (default 30.0)
        m:            angular margin in radians (default 0.5)
                      — this is what we ablate in Experiment C
    """
    def __init__(self, in_features, num_classes, s=30.0, m=0.5):
        super().__init__()
        self.s           = s
        self.m           = m
        self.weight      = nn.Parameter(
            torch.FloatTensor(num_classes, in_features)
        )
        nn.init.xavier_uniform_(self.weight)

        # Precompute margin trig values
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)
        self.mm    = math.sin(math.pi - m) * m

    def forward(self, embeddings, labels):
        # Normalize embeddings and weights → cosine similarity
        cosine = F.linear(
            F.normalize(embeddings),
            F.normalize(self.weight)
        )
        sine   = torch.sqrt(1.0 - cosine ** 2 + 1e-6)

        # Apply angular margin to the target class
        phi    = cosine * self.cos_m - sine * self.sin_m
        phi    = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot encode labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # Final logits: margin applied to target, raw cosine elsewhere
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.s

        return F.cross_entropy(logits, labels)


# IMAGE MODEL ===

class ShopeeImageModel(nn.Module):
    """
    EfficientNet backbone + embedding head.
    Swap model_name in ablation A (B0 / B3 / B5).
    """
    def __init__(self, model_name='efficientnet_b5', embed_dim=512, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )
        in_features = self.backbone.num_features
        self.embedding = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.BatchNorm1d(embed_dim)
        )

    def forward(self, x):
        features   = self.backbone(x)
        embeddings = self.embedding(features)
        return F.normalize(embeddings, dim=1)  # L2-normalize


# GET TEXT EMBEDDINGS ===

def get_text_embeddings(titles, model_name='paraphrase-xlm-r-multilingual-v1'):
    """
    Encode product titles into dense vectors.
    Swap model_name in ablation B:
      - 'tfidf'                                → sparse TF-IDF
      - 'paraphrase-multilingual-MiniLM-L12-v2'→ lightweight SBERT
      - 'paraphrase-xlm-r-multilingual-v1'     → full XLM-R (baseline)
    """
    if model_name == 'tfidf':
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf = TfidfVectorizer(max_features=25000)
        return tfidf.fit_transform(titles).toarray().astype(np.float32)

    # Sentence-Transformers path
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        titles.tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    return embeddings


# MEAN F1 SCORE ===

def get_neighbors(embeddings, threshold=0.5):
    """
    KNN retrieval: for each item find all neighbors
    with cosine similarity > threshold.
    Returns list of matched posting_id sets.
    """
    model = NearestNeighbors(n_neighbors=50, metric='cosine')
    model.fit(embeddings)
    distances, indices = model.kneighbors(embeddings)

    predictions = []
    for i in range(len(embeddings)):
        # cosine distance → cosine similarity
        idx = np.where(1 - distances[i] >= threshold)[0]
        predictions.append(indices[i][idx].tolist())
    return predictions


def row_f1(pred_set, gt_set):
    """F1 for a single row."""
    tp = len(pred_set & gt_set)
    if tp == 0:
        return 0.0
    precision = tp / len(pred_set)
    recall    = tp / len(gt_set)
    return 2 * precision * recall / (precision + recall)


def mean_f1(df, predictions):
    """
    Compute mean F1 across all postings.
    df must have columns: posting_id, label_group
    predictions: list of lists of predicted indices
    """
    # Ground truth: for each posting, which other postings share its label?
    gt = df.groupby('label_group')['posting_id'].apply(set).to_dict()
    gt_per_row = df['label_group'].map(gt)

    scores = []
    for i, pred_indices in enumerate(predictions):
        pred_ids = set(df['posting_id'].iloc[pred_indices])
        gt_ids   = gt_per_row.iloc[i]
        scores.append(row_f1(pred_ids, gt_ids))

    return np.mean(scores)


# LOGGER ===

def log_result(experiment, img_model, text_model, arcface_m, f1_score):
    """
    Append one row to results/ablation_results.csv.
    Call this at the end of every experiment notebook.
    """
    results_path = os.path.join(CFG.RESULTS_DIR, 'ablation_results.csv')
    row = pd.DataFrame([{
        'experiment'  : experiment,
        'img_model'   : img_model,
        'text_model'  : text_model,
        'arcface_m'   : arcface_m,
        'f1_score'    : round(f1_score, 4),
    }])

    if os.path.exists(results_path):
        existing = pd.read_csv(results_path)
        updated  = pd.concat([existing, row], ignore_index=True)
    else:
        updated = row

    updated.to_csv(results_path, index=False)
    print(f"✅ Logged: {experiment} → F1 = {f1_score:.4f}")
    print(updated.to_string(index=False))

# ===

import sys
sys.path.append('../src')
from utils import CFG, get_transforms, ShopeeDataset

print(f"Device: {CFG.DEVICE}")
print(f"Data dir exists: {os.path.exists(CFG.DATA_DIR)}")
print("✅ utils.py loaded successfully")