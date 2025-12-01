from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from skimage import draw
from sklearn.metrics import roc_auc_score

from new_method import predict_labels

frames_path = Path("fake_frames.npy")
features_path = Path("fake_features.npy")

frames = None  # init frames in this scope
features = None  # init frames in this scope
num_frames = 5000

# create fake "video"
if not frames_path.exists():
    mean = 0
    std_dev = 0.2

    scene = np.zeros((224, 224, 3))
    scene[25:77, 50:100, :] = np.ones(3)
    scene[100:126, 91:122, :] = np.ones(3)
    scene[200:220, 13:70, :] = np.ones(3)

    frames = []
    for i in range(num_frames):
        frame = scene.copy()
        noise = (
            np.abs(np.random.normal(loc=mean, scale=std_dev, size=(224, 224, 3))) / 10
        )
        if i >= 500 and i < 900:
            rr, cc = draw.polygon([67, 10, 119], [150, 177, 220])
            frame[rr, cc, :] = np.array([0.7, 0.1, 0.5])

        frames.append((frame + noise) / 2)

    frames = np.array(frames)

    with open(frames_path, "wb") as f:
        np.save(f, frames)
else:
    frames = np.load(frames_path)


# detect ResNet50 features
if not features_path.exists():
    from keras.applications.resnet50 import ResNet50, preprocess_input

    resnet_model = ResNet50(include_top=False, weights="imagenet", pooling="avg")

    # Convert to array and preprocess
    preprocessed = np.array(list(map(preprocess_input, frames.copy())))

    # Forward pass
    features = []

    for thing in preprocessed:
        feat = resnet_model.predict(np.array([thing]), verbose=0)[0]  # shape (2048,)
        features.append(feat)

    features = np.array(features)

    with open(features_path, "wb") as f:
        np.save(f, features)
else:
    features = np.load(features_path)


print(features.shape, frames.shape)


full_scores, full_labels = predict_labels(features)
gt_labels = np.zeros(len(full_labels))
gt_labels[500:900] = 1

print("Ground Truth Outlier Ratio:", 400 / num_frames)
print("Predicted Outlier Ratio:", np.sum(full_labels) / len(full_labels))
print("AUC Score:", roc_auc_score(gt_labels, full_scores))

plt.figure()
plt.plot(full_labels, alpha=0.25, label="Raw Output")
plt.plot(gaussian_filter1d(full_labels, 3), label="Convolved Output")
plt.plot(gt_labels, "--", label="Ground Truth")
plt.ylabel("Label")
plt.xlabel("Frame")
plt.legend()
plt.show()
