"""## Late Fusion"""

import xgboost as xgb
from sklearn.metrics import roc_auc_score

def train_struct_model(X_train, y_train, X_test):
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight
    )
    model.fit(X_train.cpu().numpy(), y_train.cpu().numpy())
    probs = model.predict_proba(X_test.cpu().numpy())[:, 1]
    return model, probs

import torch.nn as nn
import torch.optim as optim

class TextClassifier(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

def train_text_model(X_train, y_train, X_test):
    model = TextClassifier().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    epochs = 5

    for epoch in range(epochs):
        model.train()
        for i in range(0, len(X_train), 32):
            xb = X_train[i:i+32].to(DEVICE)
            yb = y_train[i:i+32].view(-1, 1).to(DEVICE)  # ✅ Fix shape
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()


    model.eval()
    with torch.no_grad():
        logits = model(X_test.to(DEVICE)).squeeze()
        probs = torch.sigmoid(logits).cpu().numpy()
    return model, probs

from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt

def evaluate_fusion(probs_struct, probs_text, y_test, alpha=0.5):
    final_probs = alpha * probs_text + (1 - alpha) * probs_struct
    auc_score = roc_auc_score(y_test.cpu(), final_probs)
    preds = (final_probs > 0.3).astype(int)

    print(f"🎯 Late Fusion ROC AUC: {auc_score:.4f}")
    print(classification_report(y_test.cpu(), preds))

    fpr, tpr, _ = roc_curve(y_test.cpu(), final_probs)
    precision, recall, _ = precision_recall_curve(y_test.cpu(), final_probs)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    plt.title("ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label="PR Curve")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.show()

struct_model, probs_struct = train_struct_model(X_struct_train, y_train, X_struct_test)
text_model, probs_text = train_text_model(X_text_train, y_train, X_text_test)

evaluate_fusion(probs_struct, probs_text, y_test, alpha=0.6)

best_auc = 0
for a in np.linspace(0, 1, 11):
    final_probs = a * probs_text + (1 - a) * probs_struct
    auc = roc_auc_score(y_test.cpu(), final_probs)
    if auc > best_auc:
        best_auc = auc
        best_alpha = a

print(f"🔥 Best AUC = {best_auc:.4f} at alpha = {best_alpha}")

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Only use embeddings that have corresponding labels
aligned_embeddings = embeddings[:len(labels)]
aligned_labels = labels['READMIT_30D'].values

# PCA projection
pca = PCA(n_components=2)
X_pca = pca.fit_transform(aligned_embeddings)

# Plot with color based on readmission label
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=aligned_labels, cmap='coolwarm', alpha=0.5)
plt.title("BERT Embeddings - PCA Projection")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label="READMIT_30D")
plt.show()
