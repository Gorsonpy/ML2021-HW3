import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataset

def get_preudo_labels(semi_batch, dataset, model, threshold = 0.65):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_loader = DataLoader(dataset, batch_size = semi_batch, shuffle = False)
    model.eval()
    # dim = -1 means the last dimension
    softmax = nn.Softmax(dim = -1)
    for batch in tqdm(data_loader):
        # batch: (img, label)
        img, _ = batch
        with torch.no_grad():
            logits = model(img.to(device))

        probs = softmax(logits)
        # construct a new dataset
        for i in range(len(probs)):
            if torch.max(probs[i]) > threshold:
                _, label = torch.max(probs[i], dim = 0)
                dataset.add_pseudo_data(img[i], label.item())

    model.train()
    return dataset


