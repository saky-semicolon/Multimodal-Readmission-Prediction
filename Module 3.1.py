"""## Model Architecture Improvement"""

class MultimodalFusionModel(nn.Module):
    def __init__(self, struct_dim, text_dim=768):
        super().__init__()
        self.struct_fc = nn.Sequential(
            nn.Linear(struct_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.text_fc = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1)
        )

    def forward(self, x_struct, x_text):
        s = self.struct_fc(x_struct)
        t = self.text_fc(x_text)
        combined = torch.cat((s, t), dim=1)
        return self.classifier(combined)

class AttentionFusionModel(nn.Module):
    def __init__(self, struct_dim, text_dim=768, hidden_dim=128):
        super().__init__()
        self.struct_fc = nn.Sequential(
            nn.Linear(struct_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        self.text_fc = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=1, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x_struct, x_text):
        s = self.struct_fc(x_struct).unsqueeze(1)
        t = self.text_fc(x_text).unsqueeze(1)
        x_cat = torch.cat((s, t), dim=1)
        attn_out, _ = self.attn(x_cat, x_cat, x_cat)
        return self.classifier(attn_out.mean(dim=1))

class HierarchicalFusionModel(nn.Module):
    def __init__(self, struct_dim, text_dim=768):
        super().__init__()
        self.struct_encoder = nn.Sequential(
            nn.Linear(struct_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1)
        )

    def forward(self, x_struct, x_text):
        struct_feat = self.struct_encoder(x_struct)
        text_feat = self.text_encoder(x_text)
        return self.fusion_head(torch.cat((struct_feat, text_feat), dim=1))

def train_model(model, name, epochs=100, batch_size=32):
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()]).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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

from sklearn.metrics import precision_recall_curve

def evaluate_model(model, name):
    model.eval()
    with torch.no_grad():
        logits = model(X_struct_test, X_text_test)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > 0.3).astype(int)
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
concat_model = MultimodalFusionModel(struct_dim=X_struct.shape[1])
concat_model = train_model(concat_model, "Concatenation")
evaluate_model(concat_model, "Concatenation")

print("\n🧠 Running Attention-Based Fusion")
attn_model = AttentionFusionModel(struct_dim=X_struct.shape[1])
attn_model = train_model(attn_model, "Attention")
evaluate_model(attn_model, "Attention")

print("\n🏗️ Running Hierarchical Fusion")
hier_model = HierarchicalFusionModel(struct_dim=X_struct.shape[1])
hier_model = train_model(hier_model, "Hierarchical")
evaluate_model(hier_model, "Hierarchical")
