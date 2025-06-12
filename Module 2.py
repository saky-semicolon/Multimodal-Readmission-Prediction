"""# Module 2

Module 2 implemented baseline predictive models using structured and unstructured data separately to establish a performance benchmark for ICU readmission prediction. For structured data, an XGBoost classifier was trained on the engineered features with class imbalance handled via scale_pos_weight. For text data, precomputed ClinicalBERT embeddings were used to train a deep classifier with weighted binary cross-entropy loss. Both models were evaluated using ROC AUC, precision, recall, and F1-score. While neither modality achieved strong standalone performance, these results validated the hypothesis that single-modality models are limited and highlighted the need for more expressive multimodal approaches.
"""

import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# === CONFIG ===
RAW_NOTES = "/kaggle/input/mmic-dataset/mimic-iii-clinical-database-1.4/NOTEEVENTS.csv/NOTEEVENTS.csv"
RAW_PATH = "/kaggle/input/mmic-dataset/mimic-iii-clinical-database-1.4/ADMISSIONS.csv/ADMISSIONS.csv"
STRUCTURED_PATH = "/kaggle/input/processed-data/Output Data/structured_features.csv"
EMBEDDING_PATH = "/kaggle/input/processed-data/Output Data/bert_embeddings.pt"
LABEL_PATH = "/kaggle/input/processed-data/Output Data/readmission_labels.csv"


# === STEP 1: Generate 30-Day Readmission Labels ===
def generate_readmission_labels(adm_path, output_path):
    df = pd.read_csv(adm_path)
    df['ADMITTIME'] = pd.to_datetime(df['ADMITTIME'])
    df['DISCHTIME'] = pd.to_datetime(df['DISCHTIME'])

    df = df.sort_values(by=['SUBJECT_ID', 'ADMITTIME'])
    df['NEXT_ADMITTIME'] = df.groupby('SUBJECT_ID')['ADMITTIME'].shift(-1)
    df['READMIT_30D'] = ((df['NEXT_ADMITTIME'] - df['DISCHTIME']).dt.days <= 30).astype(int)
    df['READMIT_30D'] = df['READMIT_30D'].fillna(0).astype(int)

    labels = df[['SUBJECT_ID', 'HADM_ID', 'READMIT_30D']]
    labels.to_csv(output_path, index=False)
    print(f"✅ Saved readmission labels: {output_path}")
    return labels

# === STEP 2: XGBoost on Structured Data ===
def run_xgboost(structured_path, label_path):
    print("🚀 Running XGBoost on structured features...")

    # Load data
    df = pd.read_csv(structured_path)
    labels = pd.read_csv(label_path)
    df = pd.merge(df, labels, on='HADM_ID')

    X = df.drop(columns=['SUBJECT_ID_x', 'HADM_ID', 'READMIT_30D'])
    y = df['READMIT_30D']

    # Handle imbalance
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    scale_pos_weight = neg / pos
    print(f"Class imbalance: scale_pos_weight = {scale_pos_weight:.2f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # Train model
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight
    )
    model.fit(X_train, y_train)

    # Predict and evaluate
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs > 0.3).astype(int)  # Lower threshold to improve recall

    from sklearn.metrics import roc_auc_score, classification_report, roc_curve, precision_recall_curve, auc
    import matplotlib.pyplot as plt

    auc_score = roc_auc_score(y_test, probs)
    print(f"\n🎯 XGBoost AUC: {auc_score:.4f}")
    print(classification_report(y_test, preds))

    # === ROC and PR plots
    fpr, tpr, _ = roc_curve(y_test, probs)
    precision, recall, _ = precision_recall_curve(y_test, probs)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("XGBoost ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()

    plt.subplot(1, 2, 2)
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.2f}")
    plt.title("XGBoost Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()

    plt.tight_layout()
    plt.savefig("xgboost_curves.png")
    plt.show()


# === STEP 3: ClinicalBERT + Classifier on Text Embeddings ===
class BERTClassifier(nn.Module):
    def __init__(self, hidden_size=768):
        super(BERTClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.classifier(x)

def run_bert_classifier(embedding_path, label_path, note_path):
    print("Loading embeddings and aligning with labels...")
    embeddings = torch.load(embedding_path)

    # Match labels with discharge summaries
    notes = pd.read_csv(note_path, low_memory=False)
    discharges = notes[notes['CATEGORY'] == 'Discharge summary'][['HADM_ID']].reset_index(drop=True)
    labels_df = pd.read_csv(label_path)
    merged = pd.merge(discharges, labels_df, on='HADM_ID')

    if len(embeddings) != len(merged):
        raise ValueError(f"Mismatch: {len(embeddings)} embeddings vs {len(merged)} labels")

    y = torch.tensor(merged['READMIT_30D'].values).float()

    # Train/test split
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, stratify=y, test_size=0.2, random_state=42)
    X_train, X_test = embeddings[train_idx], embeddings[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # === Updated classifier
    class BERTClassifier(nn.Module):
        def __init__(self, hidden_size=768):
            super(BERTClassifier, self).__init__()
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1)
            )

        def forward(self, x):
            return self.classifier(x)

    model = BERTClassifier().to("cuda" if torch.cuda.is_available() else "cpu")
    device = next(model.parameters()).device

    # === Weighted loss
    pos = (y_train == 1).sum().item()
    neg = (y_train == 0).sum().item()
    class_weight = neg / pos
    print(f"Class imbalance: pos_weight = {class_weight:.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weight]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=2e-5)

    # === Training
    epochs = 5
    print("Training BERT classifier...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i in range(0, len(X_train), 32):
            xb = X_train[i:i+32].to(device)
            yb = y_train[i:i+32].unsqueeze(1).to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}")

    # === Evaluation
    model.eval()
    with torch.no_grad():
        logits = model(X_test.to(device)).squeeze()
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > 0.3).astype(int)  # Lower threshold to improve recall

        from sklearn.metrics import roc_auc_score, classification_report, roc_curve, precision_recall_curve, auc
        import matplotlib.pyplot as plt

        auc_score = roc_auc_score(y_test, probs)
        print(f"\n🎯 ClinicalBERT AUC: {auc_score:.4f}")
        print(classification_report(y_test, preds))

        # === ROC & PR plots
        fpr, tpr, _ = roc_curve(y_test, probs)
        precision, recall, _ = precision_recall_curve(y_test, probs)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title("ClinicalBERT ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()

        plt.subplot(1, 2, 2)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.2f}")
        plt.title("ClinicalBERT Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()

        plt.tight_layout()
        plt.savefig("bert_curves.png")
        plt.show()

# === MAIN EXECUTION ===
if __name__ == "__main__":
    if not os.path.exists(LABEL_PATH):
        generate_readmission_labels(RAW_PATH, LABEL_PATH)

    run_xgboost(STRUCTURED_PATH, LABEL_PATH)
    run_bert_classifier(EMBEDDING_PATH, LABEL_PATH, RAW_NOTES)

"""**Import Note about results :**

**Your results are not good enough to deploy, but they are good enough to publish — as weak single-modality baselines that justify the need for attention-based multimodal fusion.**

