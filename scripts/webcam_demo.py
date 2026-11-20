"""Live facial emotion recognition from a webcam (or a single image).

    python scripts/webcam_demo.py --run outputs/cnn_sqrtw            # webcam, press q to quit
    python scripts/webcam_demo.py --run outputs/cnn_sqrtw --image me.jpg   # one photo

Needs the run's best.pt checkpoint (download it from Colab; it is not in git).
Faces are found with OpenCV's Haar cascade, cropped with a small margin,
converted to 48x48 grayscale and normalised exactly like the training data.
Predictions are smoothed over the last few frames so the label doesn't flicker.
"""
import argparse
import os
import sys
from collections import deque

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import CLASSES, FER_MEAN, FER_STD, IMG_SIZE  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils import get_device  # noqa: E402

COLORS = {  # BGR
    "angry": (40, 40, 220), "disgust": (40, 140, 40), "fear": (160, 60, 160), "happy": (40, 200, 240),
    "neutral": (200, 200, 200), "sad": (200, 120, 40), "surprise": (40, 160, 255),
}


def load_model(run_dir, device):
    ckpt = torch.load(os.path.join(run_dir, "best.pt"), map_location=device, weights_only=False)
    model = build_model(ckpt["model"], len(CLASSES), pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


def detect_faces(gray, cascade):
    return cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60))


@torch.no_grad()
def predict_face(model, gray, box, device, margin=0.1):
    x, y, w, h = box
    m = int(margin * w)
    x0, y0 = max(x - m, 0), max(y - m, 0)
    x1, y1 = min(x + w + m, gray.shape[1]), min(y + h + m, gray.shape[0])
    face = cv2.resize(gray[y0:y1, x0:x1], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(face).float().div(255).sub(FER_MEAN).div(FER_STD)[None, None].to(device)
    return model(t).softmax(1)[0].cpu().numpy()


def draw(frame, box, probs):
    x, y, w, h = box
    label = CLASSES[int(probs.argmax())]
    color = COLORS[label]
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, f"{label} {probs.max():.0%}", (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    # probability bars next to the face
    bx, by = x + w + 10, y
    for i, (c, p) in enumerate(zip(CLASSES, probs)):
        yy = by + i * 18
        cv2.rectangle(frame, (bx, yy), (bx + int(100 * p), yy + 12), COLORS[c], -1)
        cv2.putText(frame, c, (bx + 105, yy + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return label


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="outputs/cnn_sqrtw", help="run folder containing best.pt")
    p.add_argument("--image", help="classify faces in this image instead of the webcam")
    p.add_argument("--camera", type=int, default=0, help="webcam index")
    p.add_argument("--smooth", type=int, default=5, help="frames to average predictions over")
    a = p.parse_args()

    device = get_device()
    model = load_model(a.run, device)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    if a.image:
        frame = cv2.imread(a.image)
        if frame is None:
            sys.exit(f"Could not read {a.image}")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray, cascade)
        if len(faces) == 0:
            print("No face found; classifying the whole image as one face.")
            faces = [(0, 0, gray.shape[1], gray.shape[0])]
        for box in faces:
            probs = predict_face(model, gray, box, device)
            label = draw(frame, box, probs)
            print(f"{label}: " + ", ".join(f"{c} {q:.2f}" for c, q in zip(CLASSES, probs)))
        out = os.path.splitext(a.image)[0] + "_emotion.jpg"
        cv2.imwrite(out, frame)
        print(f"Saved {out}")
        return

    cap = cv2.VideoCapture(a.camera)
    if not cap.isOpened():
        sys.exit("Could not open the webcam. Try --camera 1, or check camera permissions.")
    history = deque(maxlen=a.smooth)
    print("Press q to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror view
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray, cascade)
        if len(faces):
            box = max(faces, key=lambda b: b[2] * b[3])  # largest face
            history.append(predict_face(model, gray, box, device))
            draw(frame, box, np.mean(history, axis=0))
        cv2.imshow("Facial Emotion Recognition (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
