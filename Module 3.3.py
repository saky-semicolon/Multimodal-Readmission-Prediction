"""## DeepResidualFusion"""

import torch.nn.functional as F

class DeepResidualFusion(nn.Module):
    def __init__(self, struct_dim, text_dim=768):
        super(DeepResidualFusion, self).__init__()

        # Structured encoder
        self.struct_net = nn.Sequential(
            nn.Linear(struct_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.GELU()
        )

        # Text encoder
        self.text_net = nn.Sequential(
            nn.Linear(text_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.GELU()
        )

        # Fusion + residual layers
        self.fusion1 = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.4)
        )
        self.fusion2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        self.output = nn.Linear(256, 1)

    def forward(self, x_struct, x_text):
        struct_out = self.struct_net(x_struct)
        text_out = self.text_net(x_text)

        fused = torch.cat([struct_out, text_out], dim=1)

        # Residual block
        fusion = self.fusion1(fused) + fused
        fusion = self.fusion2(fusion)

        return self.output(fusion)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()

def train_model(model, name, epochs=20, batch_size=64):
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i in range(0, len(y_train), batch_size):
            xb_struct = X_struct_train[i:i+batch_size]
            xb_text = X_text_train[i:i+batch_size]
            yb = y_train[i:i+batch_size]
            optimizer.zero_grad()
            preds = model(xb_struct, xb_text)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"{name} - Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}")
    return model

def evaluate_model(model, name):
    model.eval()
    with torch.no_grad():
        logits = model(X_struct_test, X_text_test)
        probs = torch.sigmoid(logits).cpu().numpy()
        from sklearn.metrics import precision_recall_curve

        # Compute best threshold using F1-score
        probs_flat = probs.ravel()
        y_true = y_test.cpu().numpy().ravel()

        precision, recall, thresholds = precision_recall_curve(y_true, probs_flat)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_threshold = thresholds[np.argmax(f1_scores)]
        print(f"📌 Optimal threshold by F1 = {best_threshold:.2f}")

        # Use this threshold instead of fixed 0.3
        preds = (probs > best_threshold).astype(int)

        y_true = y_test.cpu().numpy()

    auc_score = roc_auc_score(y_true, probs)
    print(f"\n🎯 {name} ROC AUC: {auc_score:.4f}")
    print(classification_report(y_true, preds))

    fpr, tpr, _ = roc_curve(y_true, probs)
    precision, recall, _ = precision_recall_curve(y_true, probs)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"{name} ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.2f}")
    plt.title(f"{name} Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"/kaggle/working/{name.lower().replace(' ', '_')}_curves.png")
    plt.show()

print("\n🔗 Running Concatenation-Based Fusion")
concat_model = DeepResidualFusion(struct_dim=X_struct.shape[1])
concat_model = train_model(concat_model, "Concatenation")
evaluate_model(concat_model, "Concatenation")
