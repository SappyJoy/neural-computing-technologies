import torch

BATCH_SIZE = 5  # increase / decrease according to GPU memeory
RESIZE_TO = 256  # resize the image for training and transforms
NUM_EPOCHS = 13  # number of epochs to train for
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# classes: 0 index is reserved for background
CLASSES = ['background','Yellowfin tuna']
NUM_CLASSES = len(CLASSES)
# whether to visualize images after creating the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False
# location to save model and plots
OUT_DIR = "../outputs"
SAVE_PLOTS_EPOCH = 1  # save loss plots after these many epochs
SAVE_MODEL_EPOCH = 1  # save model after these many epochs

RANDOM_SEED = 15

print(DEVICE)