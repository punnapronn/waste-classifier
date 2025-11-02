from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
from config import TRAIN_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE, OUTPUT_DIR
import os

def create_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=25,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.15
    )
    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, subset='training', class_mode='categorical', seed=42
    )
    val_gen = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, subset='validation', class_mode='categorical', seed=42
    )
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)

    # save mapping
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
        json.dump(train_gen.class_indices, f, indent=4)

    return train_gen, val_gen, test_gen
