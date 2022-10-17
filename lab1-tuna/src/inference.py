import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import glob as glob
from model import create_model
from config import CLASSES, NUM_CLASSES, DEVICE, RESIZE_TO
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image

if __name__ == "__main__":
    # load the model and the trained weights
    model = create_model(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(
        '../outputs/model13.pth', map_location=DEVICE
    ))
    model.eval()
    # directory where all the images are present
    DIR_TEST = '../resources/test'
    test_images = glob.glob(f"{DIR_TEST}/*")
    print(f"Test instances: {len(test_images)}")
    # classes: 0 index is reserved for background
    detection_threshold = 0.5

    for i in range(len(test_images)):
        # get the image file name for saving output later on
        image_name = test_images[i].split('/')[-1].split('.')[0]
        image = cv2.imread(test_images[i])
        image_resized = cv2.resize(image, (RESIZE_TO, RESIZE_TO))
        orig_image = image.copy()
        # BGR to RGB
        image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        # make the pixel range between 0 and 1
        image_resized /= 255.0
        # bring color channels to front
        image_resized = np.transpose(image_resized, (2, 0, 1)).astype(np.float)
        # convert to tensor
        image_resized = torch.tensor(image_resized, dtype=torch.float)
        # add batch dimension
        image_resized = torch.unsqueeze(image_resized, 0)
        with torch.no_grad():
            outputs = model(image_resized)

        # load all detection to CPU for further operations
        print(outputs)
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        # carry further only if there are detected boxes
        if len(outputs[0]['boxes']) != 0:
            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()
            # filter out boxes according to `detection_threshold`
            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            draw_boxes = boxes.copy()
            # get all the predicited class names
            pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]

            # draw the bounding boxes and write the class name on top of it
            for j, box in enumerate(draw_boxes):
                cv2.rectangle(orig_image,
                              (int(box[0]), int(box[1])),
                              (int(box[2]), int(box[3])),
                              (0, 0, 255), 2)
                cv2.putText(orig_image, pred_classes[j],
                            (int(box[0]), int(box[1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                            2, lineType=cv2.LINE_AA)
            cv2.imshow('Prediction', orig_image)
            cv2.waitKey(1)
            cv2.imwrite(f"../test_predictions/{image_name}.jpg", orig_image, )
        print(f"Image {i + 1} done...")
        print('-' * 50)
    print('TEST PREDICTIONS COMPLETE')
    cv2.destroyAllWindows()