from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2

def default_loader(path):
    # return Image.open(path).convert('RGB')
    return Image.open(path).convert('L')
class MyDataset(Dataset):
    def __init__(self, txt, transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[1], int(words[0])))
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_path, label = self.imgs[index]
        img = self.loader(img_path)
        img = img.resize((32, 32))
        table = []
        for i in range(256):
            if i < 200:
                table.append(1)
            else:
                table.append(0)
        photo = img.point(table, '1')
        photo.save('test.jpg')
        if self.transform is not None:
            img = self.transform(photo)
        return img, label

    def __len__(self):
        return len(self.imgs)