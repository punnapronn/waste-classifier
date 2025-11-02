import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from src.data_utils import create_generators
from src.config import OUTPUT_DIR, MODEL_DIR

# ===============================
# 🔧 Setup
# ===============================
os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)

# โหลด test data
_, _, test_gen = create_generators()

# โหลด model ที่ดีที่สุด
model_path = os.path.join(MODEL_DIR, "best_model.h5")
model = load_model(model_path)

# ===============================
# 📊 Evaluate Model Performance
# ===============================
print("🔹 Evaluating model on test set...")
test_loss, test_acc = model.evaluate(test_gen, verbose=1)
print(f"\n✅ Test Accuracy: {test_acc:.4f}")
print(f"✅ Test Loss: {test_loss:.4f}\n")

# ===============================
# 🔍 Classification Report
# ===============================
preds = model.predict(test_gen)
y_pred = np.argmax(preds, axis=1)
y_true = test_gen.classes
class_names = list(test_gen.class_indices.keys())

# สร้างรายงาน Precision / Recall / F1
report = classification_report(
    y_true, y_pred, target_names=class_names, output_dict=True
)
report_path = os.path.join(OUTPUT_DIR, "report.json")
with open(report_path, "w") as f:
    json.dump(report, f, indent=4)
print(f"📄 Classification report saved at: {report_path}")

# ===============================
# 🔲 Confusion Matrix
# ===============================
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
cm_path = os.path.join(OUTPUT_DIR, "figures", "confusion_matrix.png")
plt.savefig(cm_path, bbox_inches='tight')
plt.close()
print(f"🖼️ Confusion matrix saved at: {cm_path}")

# ===============================
# 📈 Optional: Visualize Training History
# ===============================
history_path = os.path.join(OUTPUT_DIR, "history_v2.json")
if os.path.exists(history_path):
    with open(history_path, "r") as f:
        history = json.load(f)

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history["accuracy"], label="train_acc")
    plt.plot(history["val_accuracy"], label="val_acc")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history["loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    hist_fig_path = os.path.join(OUTPUT_DIR, "figures", "training_history.png")
    plt.savefig(hist_fig_path, bbox_inches='tight')
    plt.close()
    print(f"📊 Training history plot saved at: {hist_fig_path}")

# ===============================
# 🧩 Summary
# ===============================
print("\n🔹 Summary of Metrics per Class:")
for cls, metrics in report.items():
    if isinstance(metrics, dict):
        print(f"{cls:15s} "
              f"Precision={metrics['precision']:.2f}, "
              f"Recall={metrics['recall']:.2f}, "
              f"F1-score={metrics['f1-score']:.2f}")

print("\n✅ Evaluation complete!")
