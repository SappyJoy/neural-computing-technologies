import cv2
from config import BATCH_SIZE
from FishnetDataset import FishnetDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import collate_fn

train_dataset = FishnetDataset(
    annotations_file="../resources/labels/foid_labels_bbox_v012.csv", img_dir="../resources/images"
)
valid_dataset = FishnetDataset(
    annotations_file="../resources/labels/foid_labels_bbox_v012.csv",
    img_dir="../resources/images",
    transform=transforms.Resize([256, 256]),
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(valid_dataset)}\n")

if __name__ == "__main__":
    # sanity check of the Dataset pipeline with sample visualization
    print(f"Number of training images: {len(train_dataset)}")

    # function to visualize a single sample
    def visualize_sample(image, target):
        box = target["boxes"][0]
        label = target["labels"][0]
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 1)
        cv2.putText(image, label, (int(box[0]), int(box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Image", image)
        cv2.waitKey(0)

    NUM_SAMPLES_TO_VISUALIZE = 5
    for i in range(NUM_SAMPLES_TO_VISUALIZE):
        image, target = train_dataset[i]
        visualize_sample(image, target)
