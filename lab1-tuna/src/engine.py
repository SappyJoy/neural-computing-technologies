import time

# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim.lr_scheduler import StepLR
from tqdm.auto import tqdm

from config import DEVICE, NUM_CLASSES, NUM_EPOCHS, OUT_DIR
from config import SAVE_PLOTS_EPOCH, SAVE_MODEL_EPOCH
from datasets import train_loader, valid_loader
from model import create_model
from utils import Averager

plt.style.use('ggplot')

classLossFunc = CrossEntropyLoss()
bboxLossFunc = MSELoss()

if __name__ == '__main__':
    # initialize the model and move to the computation device
    model = create_model(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    # get the model parameters
    params = [p for p in model.parameters() if p.requires_grad]
    # define the optimizer
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.01)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    # initialize the Averager class
    train_loss_hist = Averager()
    val_loss_hist = Averager()
    train_itr = 1
    val_itr = 1
    # train and validation loss lists to store loss values of all...
    # ... iterations till ena and plot graphs for all iterations
    train_loss_list = []
    val_loss_list = []
    # name to save the trained model with
    MODEL_NAME = 'model'
    # whether to show transformed images from data loader or not
    # start the training epochs
    for epoch in range(NUM_EPOCHS):
        print(f"\nEPOCH {epoch + 1} of {NUM_EPOCHS}")
        # reset the training and validation loss histories for the current epoch
        train_loss_hist.reset()
        val_loss_hist.reset()
        # create two subplots, one for each, training and validation
        figure_1, train_ax = plt.subplots()
        figure_2, valid_ax = plt.subplots()
        # start timer and carry out training and validation
        start = time.time()

        prog_bar_train = tqdm(train_loader, total=len(train_loader))

        for i, data in enumerate(prog_bar_train):
            images, targets = data


            images =images.to(DEVICE)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images)

            bboxLoss = bboxLossFunc(loss_dict[0], targets['boxes'])
            classLoss = classLossFunc(loss_dict[1], targets['labels'])

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            train_loss_list.append(loss_value)
            train_loss_hist.send(loss_value)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            train_itr += 1

            # update the loss value beside the progress bar for each iteration
            prog_bar_train.set_description(desc=f"Loss: {loss_value:.4f}")

        prog_bar_val = tqdm(valid_loader, total=len(valid_loader))

        for i, data in enumerate(prog_bar_val):
            images, targets = data

            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            with torch.no_grad():
                loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            val_loss_list.append(loss_value)
            val_loss_hist.send(loss_value)
            val_itr += 1
            # update the loss value beside the progress bar for each iteration
            prog_bar_val.set_description(desc=f"Loss: {loss_value:.4f}")

        scheduler.step()

        print(f"Epoch #{epoch} train loss: {train_loss_hist.value:.3f}")
        print(f"Epoch #{epoch} validation loss: {val_loss_hist.value:.3f}")
        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")
        if (epoch + 1) % SAVE_MODEL_EPOCH == 0:  # save model after every n epochs
            torch.save(model.state_dict(), f"{OUT_DIR}/model{epoch + 1}.pth")
            print('SAVING MODEL COMPLETE...\n')

        if (epoch + 1) % SAVE_PLOTS_EPOCH == 0:  # save loss plots after n epochs
            train_ax.plot(train_loss_list, color='blue')
            train_ax.set_xlabel('iterations')
            train_ax.set_ylabel('train loss')
            valid_ax.plot(val_loss_list, color='red')
            valid_ax.set_xlabel('iterations')
            valid_ax.set_ylabel('validation loss')
            figure_1.savefig(f"{OUT_DIR}/train_loss_{epoch + 1}.png")
            figure_2.savefig(f"{OUT_DIR}/valid_loss_{epoch + 1}.png")
            print('SAVING PLOTS COMPLETE...')

        if (epoch + 1) == NUM_EPOCHS:  # save loss plots and model once at the end
            train_ax.plot(train_loss_list, color='blue')
            train_ax.set_xlabel('iterations')
            train_ax.set_ylabel('train loss')
            valid_ax.plot(val_loss_list, color='red')
            valid_ax.set_xlabel('iterations')
            valid_ax.set_ylabel('validation loss')
            figure_1.savefig(f"{OUT_DIR}/train_loss_{epoch + 1}.png")
            figure_2.savefig(f"{OUT_DIR}/valid_loss_{epoch + 1}.png")
            torch.save(model.state_dict(), f"{OUT_DIR}/model{epoch + 1}.pth")

        plt.close('all')
        time.sleep(5)
