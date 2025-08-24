import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import mediapipe as mp

# --- SmallCNN (from training) ---
class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def resnet18_1ch(num_classes=10):
    m = models.resnet18(weights=None)
    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def run_webcam_with_hands(model_path, use_resnet=False, img_size=96, label_names=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    if use_resnet:
        model = resnet18_1ch(10).to(device)
    else:
        model = SmallCNN(10).to(device)

    ckpt = torch.load(model_path, map_location=device,weights_only = False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Preprocessing (same as eval_tfms)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    if label_names is None:
        label_names = [str(i) for i in range(10)]

    # Mediapipe hands setup
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=1,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get bounding box
                h, w, c = frame.shape
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                xmin, xmax = int(min(x_coords) * w), int(max(x_coords) * w)
                ymin, ymax = int(min(y_coords) * h), int(max(y_coords) * h)

                # Add padding
                pad = 20
                xmin, xmax = max(0, xmin-pad), min(w, xmax+pad)
                ymin, ymax = max(0, ymin-pad), min(h, ymax+pad)

                # Crop and preprocess
                hand_img = frame[ymin:ymax, xmin:xmax]
                if hand_img.size != 0:
                    gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
                    pil_img = Image.fromarray(gray)
                    img = transform(pil_img).unsqueeze(0).to(device)

                    with torch.no_grad():
                        logits = model(img)
                        probs = torch.softmax(logits, dim=1)[0]
                        pred = logits.argmax(1).item()
                        pred_label = label_names[pred]
                        pred_conf = probs[pred].item()

                    # Draw bounding box + label
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(frame, f"{pred_label} ({pred_conf*100:.1f}%)",
                                (xmin, ymin-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Optionally draw landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("LeapGest Recognition (with Hand Detection)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Use either SmallCNN or ResNet18
    # MODEL_PATH = r"C:\Programming\Prodigy Infotech\PRODIGY_ML_04\best_smallcnn_96.pth"
    # run_webcam_with_hands(MODEL_PATH, use_resnet=False, img_size=96)

    # For ResNet18:
    MODEL_PATH = r"C:\Programming\Prodigy Infotech\PRODIGY_ML_04\best_resnet18_96.pth"
    run_webcam_with_hands(MODEL_PATH, use_resnet=True, img_size=96)
