import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

from dataloader import extract_resnet50_feature
from new_method import predict_labels

video_path = "WIN_20251129_15_38_02_Pro.mp4"  # <-- your video file
os.makedirs("./data/StreamDataVideo", exist_ok=True)

features = []

if not os.path.exists("real_features.npy"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    filenames = []

    frame_count = 0
    max_frames = 2200  # or however many frames you want to sample

    # may have to delete this line; it was used to prevent processing on too many images (very time consuming)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 12400)

    while frame_count < max_frames:
        ret, frame = cap.read()

        if not ret:
            print("End of video or failed to read frame.")
            break

        # Save frame
        timestamp = int(time.time() * 1000)
        filepath = os.path.join(
            "./data/StreamDataVideo", f"frame_{frame_count}_{timestamp}.jpg"
        )

        try:
            cv2.imwrite(filepath, frame)
        except Exception as e:
            print(f"Failed to save frame {frame_count}: {e}")
            continue

        print(f"Saved {filepath}")

        # Extract features
        try:
            feat = extract_resnet50_feature(filepath)
            features.append(feat)
            filenames.append(filepath)
        except Exception as e:
            print(f"Failed to extract features for {filepath}: {e}")

        frame_count += 1

    cap.release()

    features = np.array(features)

    np.save("real_features.npy", np.array(features))
    np.save("filenames2.npy", np.array(filenames))
else:
    features = np.load("real_features.npy")[:200, :]


# human generated ground truth annotations. Need to manually set if using custom video
gt_labels = np.concatenate([np.zeros(200), np.ones(100)])

score_test, full_labels = predict_labels(features)

plt.figure()
plt.plot(full_labels, alpha=0.25, label="Raw Output")
plt.plot(gaussian_filter1d(full_labels, 3), label="Convolved Output")
plt.plot(gt_labels, "--", label="Ground Truth")
plt.ylabel("Label")
plt.xlabel("Frame")
plt.legend()
plt.show()
