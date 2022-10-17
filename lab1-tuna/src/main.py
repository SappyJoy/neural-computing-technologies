# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# import torch
# import torch.optim as optim
# import torchvision.ops
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter  # TensorBoard support
# from torchvision import transforms
# from torchvision.models.detection.ssd import SSD300_VGG16_Weights, ssd300_vgg16
#
# from FishnetDataset import FishnetDataset
#
# if __name__ == "__main__":
#     tb = SummaryWriter(comment="Run simple cnn on mnist")
#
#     # Device configuration
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # device = torch.device("cpu")  # My GeForce 660 isn't supported because it's too old
#     print(device)
#
#     train_data = FishnetDataset(
#         annotations_file="../resources/labels/foid_labels_bbox_v012.csv",
#         img_dir="../resources/images",
#         transform=transforms.Resize([256, 256]),
#     )
#
#     figure = plt.figure(figsize=(10, 8))
#     cols, rows = 2, 2
#     for i in range(1, cols * rows + 1):
#         sample_idx = torch.randint(len(train_data), size=(1,)).item()
#         img, target = train_data[sample_idx]
#
#         figure.add_subplot(rows, cols, i)
#         plt.title(target["labels"][0])
#         plt.axis("off")
#         plt.imshow(img.permute(1, 2, 0))
#     plt.show()
#
#     train_loader = DataLoader(
#         train_data,
#         batch_size=100,
#         shuffle=True,
#     )
#
#     test_loader = DataLoader(train_data, batch_size=100, shuffle=True)
#
#     # cnn = models.alexnet(num_classes=10).to(device)
#     model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
#     iou_loss = torchvision.ops.complete_box_iou_loss
#     optimizer = optim.Adam(model.parameters(), lr=0.0015)
#
#     print(model)
#
#     epochs = 10
#     steps = 0
#     print_every = 100
#     trainLoss = []
#     testLoss = []
#
#     for e in range(epochs):
#         running_loss = 0
#
#         for images, targets in train_loader:
#             # Forward pass
#             steps += 1
#             optimizer.zero_grad()
#
#             # images = (images.view(-1,1,28,28)).type(torch.DoubleTensor)
#             # images = np.repeat(images, 3, axis=1)
#
#             log_ps = model(images.type(torch.FloatTensor).to(device), targets.type(torch.FloatTensor).to(device))
#
#             # Loss calculation
#             # targets = targets.to(device)
#             # iou_loss()
#             # loss = loss_func(log_ps, targets)
#             # tb.add_scalar('Loss', loss, e)
#             #
#             # # Backward pass
#             # loss.backward()
#             # optimizer.step()
#             # running_loss += loss.item()
#             #
#             # # Validation step
#             # if steps % print_every == 0:
#             #     accuracy, test_loss = 0, 0
#             #
#             #     # Set mode withou gradient calculation
#             #     with torch.no_grad():
#             #         cnn.eval()
#             #
#             #         # Run validation on test dataset
#             #         for images, targets in test_loader:
#             #             # images = (images.view(-1,1,28,28)).type(torch.DoubleTensor)
#             #             # images = np.repeat(images, 3, axis=1)
#             #
#             #             log_ps = cnn(images.type(torch.FloatTensor).to(device))
#             #             targets = targets.to(device)
#             #             testsample_loss = loss_func(log_ps, targets)
#             #             test_loss += testsample_loss
#             #
#             #             tb.add_scalar('Test Loss', testsample_loss, e)
#             #
#             #             ps = torch.exp(log_ps)
#             #
#             #             top_p, top_class = ps.topk(1, dim=1)
#             #
#             #             accuracy += accuracy_score(targets.view(*top_class.shape).cpu(), top_class.cpu())
#             #
#             #     # Returnn train mode
#             #     cnn.train()
#             #
#             #     trainLoss.append(running_loss / len(train_loader))
#             #     testLoss.append(test_loss / len(test_loader))
#             #     testAccur = accuracy / len(test_loader)
#             #
#             #     tb.add_scalar('Accuracy', testAccur, e)
#             #
#             #     print("Epoch: {}/{}.. ".format(e + 1, epochs),
#             #           "Test Accuracy: {:.3f}".format(testAccur))
