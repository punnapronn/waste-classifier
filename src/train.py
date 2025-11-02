from src.data_utils import create_generators
from src.model import build_transfer_model
from src.config import MODEL_DIR, EPOCHS
import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import json

train_gen, val_gen, test_gen = create_generators()
num_classes = train_gen.num_classes
model = build_transfer_model(num_classes)

os.makedirs(MODEL_DIR, exist_ok=True)
ckpt = ModelCheckpoint(os.path.join(MODEL_DIR,"best_model.h5"), monitor='val_accuracy', save_best_only=True)
es = EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=[ckpt, es, rlr])
# save history
with open('outputs/history.json','w') as f:
    json.dump(history.history, f)
