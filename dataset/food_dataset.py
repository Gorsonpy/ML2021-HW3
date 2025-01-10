
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import DatasetFolder
from torchvision.transforms import transforms
from PIL import Image

train_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    # data argumentation needs to be added
    transforms.ToTensor(),
])

test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    # data argumentation needs to be added
    transforms.ToTensor(),
])

def get_DataFolder(path, mode='train'):
    if mode == 'train':
        return DatasetFolder(path, loader= lambda x:Image.open(x), extensions='jpg', transform=train_tfm)
    else:
        return DatasetFolder(path, loader= lambda x:Image.open(x), extensions='jpg', transform=test_tfm)

def get_food_loader(batch_size, num_workers=8, pin_memory=True, mode='train'):
    if mode == 'train':
        dataset = get_DataFolder('./training/labeled', mode)
    else:
        if mode == 'test':
            dataset =  get_DataFolder('./testing', mode)
        else:
            dataset =  get_DataFolder('./validation', mode)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=(mode=='train'))
