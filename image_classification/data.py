import torch
import os, glob
import random, csv
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from PIL import Image

class Pokemon(Dataset):
    def __init__(self, root, resize, mode):
        super(Pokemon, self).__init__()
        self.resize = resize
        self.root = root
        self.name2label = {}
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root, name)):
                continue
            self.name2label[name] = len(self.name2label.keys())
        # print(self.name2label)
        """
        {'bulbasaur': 0, 'charmander': 1, 'mewtwo': 2, 'pikachu': 3, 'squirtle': 4}
        """
        self.images, self.labels = self.load_csv('images.csv')

        if mode == 'train':
            self.images = self.images[:int(0.6*len(self.images))]
            self.labels = self.labels[:int(0.6*len(self.labels))]
        elif mode == 'val':
            self.images = self.images[int(0.6*len(self.images)):int(0.8*len(self.images))]
            self.labels = self.labels[int(0.6*len(self.labels)):int(0.8*len(self.labels))]
        else:
            self.images = self.images[int(0.8*len(self.images)):]
            self.labels = self.labels[int(0.8*len(self.labels)):]

    def load_csv(self, filename):
        images, labels = [], []
        
        for name in self.name2label.keys():
            images += glob.glob(os.path.join(self.root, name, '*.png'))
            images += glob.glob(os.path.join(self.root, name, '*.jpg'))
            images += glob.glob(os.path.join(self.root, name, '*.jpeg'))
            labels += [self.name2label[name]] * (len(images) - len(labels))
            # Add new labels only for the newly added images

        # print(len(images), len(labels))
        # 1167 1167

        combined = list(zip(images, labels))
        random.shuffle(combined)
        images[:], labels = zip(*combined)
        images, labels = list(images), list(labels)

        with open(os.path.join(self.root, filename), mode='w', newline='') as f:
            writer = csv.writer(f)
            for image, label in zip(images, labels):
                writer.writerow([image, label])

        images = []
        labels = []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img, label = row
                images.append(img)
                labels.append(int(label))

        return images, labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]
        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            transforms.Resize((int(self.resize*1.25), int(self.resize*1.25))),
            transforms.RandomRotation(15),
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        img = tf(img)
        label = torch.tensor(label)
        return img, label
    
    def denormalize(self, x_hat):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # x_hat = (x - mean) / std
        # x = x_hat * std + mean
        # x: [c, h, w]
        # mean: [3] => [3, 1, 1]
        # std: [3] => [3, 1, 1]
        mean = torch.tensor(mean).view(3, 1, 1)
        std = torch.tensor(std).view(3, 1, 1)
        print(mean.shape, std.shape)
        x = x_hat * std + mean
        return x
    
if __name__ == '__main__':
    import visdom
    import time

    viz = visdom.Visdom()
    db = Pokemon('data/pokemon', 224, 'val')
    x, y = next(iter(db))
    print('sample:', x.shape, y.shape, y)

    viz.image(db.denormalize(x), win='sample_x', opts=dict(title='sample_x'))

    loader = DataLoader(db, batch_size=32, shuffle=True)

    for x, y in loader:
        viz.images(db.denormalize(x), nrow=8, win='batch_x', opts=dict(title='batch_x'))
        viz.text(str(y.numpy()), win='batch_y', opts=dict(title='batch_y'))

        time.sleep(10)

