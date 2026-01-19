import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

VIDEO_PATH = 'C:/Users/kalha/Downloads/MSC@DBS/Research/Videos/1.mp4'

FRAMES_DIR = "C:/Users/kalha/Downloads/MSC@DBS/Research/Frames2"

AD_START_SEC = 164
AD_END_SEC = 190

def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}. Defaulting to 30 FPS.")
        return 30.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"FPS of the video: {fps}")
    return fps

def load_frame_dataset_with_features(folder_path, fps, img_size=(224, 224)):
    images = []
    labels = []
    pixel_diffs = []
    hist_diffs = []
    rolling_means = []
    rolling_stds = []

    frames = sorted(os.listdir(folder_path))
    total_frames = len(frames)

    print(f"--- Starting: Reading {total_frames} frames from directory... ---")

    for i in range(1, len(frames)):
        fpath_prev = os.path.join(folder_path, frames[i-1])
        fpath_curr = os.path.join(folder_path, frames[i])

        img_prev = load_img(fpath_prev, target_size=img_size)
        img_curr = load_img(fpath_curr, target_size=img_size)

        arr_prev = img_to_array(img_prev)
        arr_curr = img_to_array(img_curr)

        arr_prev_pre = preprocess_input(arr_prev.copy())
        arr_curr_pre = preprocess_input(arr_curr.copy())

        images.append(arr_curr_pre)
        
        # Feature Calculation
        pixel_diff = np.abs(arr_curr - arr_prev).mean()
        pixel_diffs.append(pixel_diff)

        hist_prev = cv2.calcHist([arr_prev], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
        hist_curr = cv2.calcHist([arr_curr], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
        hist_diff = cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_CHISQR)
        hist_diffs.append(hist_diff)

        if i >= 4:
            rolling_mean = np.mean(pixel_diffs[i-4:i+1])
            rolling_std = np.std(pixel_diffs[i-4:i+1])
        else:
            rolling_mean = np.mean(pixel_diffs[:i+1])
            rolling_std = np.std(pixel_diffs[:i+1])

        rolling_means.append(rolling_mean)
        rolling_stds.append(rolling_std)

        current_time_sec = i / fps
        
        if AD_START_SEC <= current_time_sec <= AD_END_SEC:
            labels.append(1) # Class 1: Ad
        else:
            labels.append(0) # Class 0: Main

    images = np.array(images)
    labels = np.array(labels)
    features = np.stack([pixel_diffs, hist_diffs, rolling_means, rolling_stds], axis=1)
    
    print(f"--- Data Loading Complete. {len(images)} samples created. ---")
    return images, features, labels

video_fps = get_video_fps(VIDEO_PATH)

X, features, y = load_frame_dataset_with_features(FRAMES_DIR, fps=video_fps)

print(f"\nData successfully loaded. X shape: {X.shape}. Class distribution: {Counter(y)}")
print("Legend: 0 = Main, 1 = Ad")

print("\n--- Starting: Train-Test Split (test_size=0.2) ---")

X_train, X_test, features_train, features_test, y_train, y_test = train_test_split(
    X, features, y, test_size=0.2, random_state=42
)

print(f"Split Complete. Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# One-hot encoding for 2 classes
y_train_cat = tf.keras.utils.to_categorical(y_train, 2)
y_test_cat = tf.keras.utils.to_categorical(y_test, 2)
print("One-hot encoding complete.")

print("\n--- Starting: Class Weight Calculation ---")
classes_in_train = np.unique(y_train)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes_in_train,
    y=y_train
)
cw = {cls: weight for cls, weight in zip(classes_in_train, class_weights)}
print("Class weights calculated:", cw)

print("\n--- Starting: Model Definition ---")
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10, width_shift_range=0.1, height_shift_range=0.1,
    zoom_range=0.15, horizontal_flip=True, fill_mode="nearest"
)

def build_model():
    base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base.trainable = False 

    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(2, activation='softmax')(x)

    model = Model(inputs=base.input, outputs=output)
    return model

model = build_model()
model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
print("Model built and compiled for 2 classes (Main vs Ad).")

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=1),
    ModelCheckpoint("best_promo_classifier.h5", save_best_only=True, monitor='val_accuracy')
]

history = model.fit(
    datagen.flow(X_train, y_train_cat, batch_size=32),
    validation_data=(X_test, y_test_cat),
    epochs=20,
    class_weight=cw,
    callbacks=callbacks
)

# Fine-tuning phase
for layer in model.layers:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-5), loss="categorical_crossentropy", metrics=["accuracy"])

history2 = model.fit(
    datagen.flow(X_train, y_train_cat, batch_size=32),
    validation_data=(X_test, y_test_cat),
    epochs=20,
    class_weight=cw,
    callbacks=callbacks
)

print("\n--- Starting: Final Model Evaluation ---")
loss, acc = model.evaluate(X_test, y_test_cat)
print(f"\nFinal Test Accuracy: {acc:.4f}")

probs = model.predict(X_test)
pred_labels = np.argmax(probs, axis=1)

cm = confusion_matrix(y_test, pred_labels)
print("\nConfusion Matrix:\n", cm)

print("\nClassification Report (Metrics for 2 Classes):")
print(classification_report(y_test, pred_labels, labels=[0, 1], target_names=["Main", "Ad"], zero_division=0))

print("--- Saving Final Model ---")
model.save("promo_classifier_final.h5")
print("Model saved as promo_classifier_final.h5")
print("--- Execution Complete ---")