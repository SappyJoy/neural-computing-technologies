import os
import random

from torch.utils.data import DataLoader

from FishnetDataset import FishnetDataset
from config import BATCH_SIZE, RANDOM_SEED
from utils import collate_fn, get_train_transform, get_valid_transform

random.seed(RANDOM_SEED)

train_dataset = FishnetDataset(
    annotations_file="../resources/labels/foid_labels_bbox_v012.csv", img_dir="../resources/images",
    transform=get_train_transform()
)
valid_dataset = FishnetDataset(
    annotations_file="../resources/labels/foid_labels_bbox_v012_validate.csv",
    img_dir="../resources/images",
    transform=get_valid_transform(),
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)#, num_workers=os.cpu_count())#, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)#, num_workers=os.cpu_count())#, collate_fn=collate_fn)
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(valid_dataset)}\n")

# if __name__ == "__main__":
#     # sanity check of the Dataset pipeline with sample visualization
#     print(f"Number of training images: {len(train_dataset)}")


#     # function to visualize a single sample
#     def visualize_sample(image, target):
#         image_ = tensor_to_image(image)
#         for i in range(len(target["boxes"])):
#             box = target["boxes"][i]
#             label = target["labels"][i]
#             img1 = ImageDraw.Draw(image_)
#             img1.rectangle([int(box[0]), int(box[1]), int(box[2]), int(box[3])],
#                            outline="green")

#             img1.text((int(box[2]), int(box[1])), CLASSES[label], (255, 255, 255))

#         image_.show()


#     NUM_SAMPLES_TO_VISUALIZE = 5
#     for i in range(NUM_SAMPLES_TO_VISUALIZE):
#         indx = random.randint(0, len(train_dataset))
#         print(indx)
#         image_, target = train_dataset[indx]
#         print(target["labels"][0])
#         visualize_sample(image_, target)
