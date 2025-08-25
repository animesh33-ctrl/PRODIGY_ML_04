import os
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image


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


def classify_images(model_path, folder_path, use_resnet=False, img_size=96, label_names=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    if use_resnet:
        model = resnet18_1ch(10).to(device)
    else:
        model = SmallCNN(10).to(device)

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    if label_names is None:
        label_names = [str(i) for i in range(10)]

    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(folder_path, filename)


            img = Image.open(img_path).convert("L")
            tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)[0]
                pred = logits.argmax(1).item()
                pred_label = label_names[pred]
                pred_conf = probs[pred].item()


            img_cv = cv2.imread(img_path)
            cv2.putText(img_cv, f"{pred_label} ({pred_conf*100:.1f}%)",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

            cv2.imshow("Hand Gesture Classification", img_cv)
            


            key = cv2.waitKey(0)
            if key & 0xFF == ord("q"):  
                break

    cv2.destroyAllWindows()



if __name__ == "__main__":
    # # For SmallCNN
    # MODEL_PATH = r"C:\Programming\Prodigy Infotech\PRODIGY_ML_04\best_smallcnn_96.pth"
    # classify_images(MODEL_PATH, r"./images", use_resnet=False, img_size=96)

    # For ResNet18
    MODEL_PATH = r"C:\Programming\Prodigy Infotech\PRODIGY_ML_04\best_resnet18_96.pth"
    classify_images(MODEL_PATH, r"./images", use_resnet=True, img_size=96)
