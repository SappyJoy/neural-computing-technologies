import torch

BATCH_SIZE = 4  # increase / decrease according to GPU memeory
RESIZE_TO = 512  # resize the image for training and transforms
NUM_EPOCHS = 100  # number of epochs to train for
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# training images and XML files directory
TRAIN_DIR = "../Microcontroller Detection/train"
# validation images and XML files directory
VALID_DIR = "../Microcontroller Detection/test"
# classes: 0 index is reserved for background
CLASSES = [
    "ALB",
    "BET",
    "BILL",
    "BILL",
    "OTH",
    "HUMAN",
    "BILL",
    "OTH",
    "DOL",
    "OTH",
    "NoF",
    "OTH",
    "LAG",
    "PLS",
    "OTH",
    "OTH",
    "OIL",
    "SHARK",
    "BILL",
    "OTH",
    "SKJ",
    "OTH",
    "BILL",
    "BILL",
    "SHARK",
    "OTH",
    "OTH",
    "YFT",
]
NUM_CLASSES = len(CLASSES)
# whether to visualize images after creating the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False
# location to save model and plots
OUT_DIR = "../outputs"
SAVE_PLOTS_EPOCH = 2  # save loss plots after these many epochs
SAVE_MODEL_EPOCH = 2  # save model after these many epochs

RANDOM_SEED = 30
