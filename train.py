from operator import concat

import torch
from PIL.Image import Image
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import DatasetFolder
from tqdm.asyncio import tqdm

from dataset.food_dataset import get_food_loader
from dataset.food_dataset import get_DataFolder
import dataset.food_dataset
from model_cnn import CNN
from semi_supervise import get_preudo_labels

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CNN().to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)
num_epochs = 80

do_semi = False
batch_size = 64
train_set = get_DataFolder('./training/labeled', batch_size)

for epoch in range(num_epochs):
    if do_semi:
        # Semi-supervised learning
        # 1. Train the model on labeled data
        # 2. Use the model to predict labels on unlabeled data
        # 3. Add the pseudo-labeled data to the labeled data
        # 4. Go to 1

        unlabeled_set = get_DataFolder('./trainging/unlabeled', mode = 'train')
        preudo_set = get_preudo_labels(batch_size, unlabeled_set, model)

        concat_dataset = ConcatDataset([train_set, preudo_set])
        train_loader = DataLoader(concat_dataset, batch_size=batch_size, shuffle=True)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    model.train()
    train_loss = []
    acc_loss = []

    for batch in tqdm(train_loader):
        images, labels = batch
        logits = model(images.to(device))
        loss = criterion(logits, labels.to(device))

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        train_loss.append(loss.item())
        acc_loss.append((logits.argmax(dim = -1) == labels.to(device)).float().mean())

    train_loss = sum(train_loss) / len(train_loss)
    acc_loss = sum(acc_loss) / len(acc_loss)

    print(f"[ Train | {epoch + 1:03d}/{num_epochs:03d} ] loss = {train_loss:.5f}, acc = {acc_loss:.5f}")

    model.eval()
    valid_loss = []
    valid_acc = []
    valid_loader = get_food_loader(batch_size, mode='valid')

    for batch in tqdm(valid_loader):
        images, labels = batch
        with torch.no_grad():
            logits = model(images.to(device))

        loss = criterion(logits, labels.to(device))
        valid_loss.append(loss.item())
        valid_acc.append((logits.argmax(dim=-1) == labels.to(device)).float().mean())

    valid_acc = sum(valid_acc) / len(valid_acc)
    valid_loss = sum(valid_loss) / len(valid_loss)

    print(f"[ Valid | {epoch + 1:03d}/{num_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")


# test save
# Make sure the model is in eval mode.
# Some modules like Dropout or BatchNorm affect if the model is in training mode.
model.eval()

# Initialize a list to store the predictions.
predictions = []
test_loader = get_food_loader(batch_size, mode='test')
# Iterate the testing set by batches.
for batch in tqdm(test_loader):
    # A batch consists of image data and corresponding labels.
    # But here the variable "labels" is useless since we do not have the ground-truth.
    # If printing out the labels, you will find that it is always 0.
    # This is because the wrapper (DatasetFolder) returns images and labels for each batch,
    # so we have to create fake labels to make it work normally.
    imgs, labels = batch

    # We don't need gradient in testing, and we don't even have labels to compute loss.
    # Using torch.no_grad() accelerates the forward process.
    with torch.no_grad():
        logits = model(imgs.to(device))

    # Take the class with greatest logit as prediction and record it.
    predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

