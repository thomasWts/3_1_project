import torch
from torch import nn, optim
from d2l import torch as d2l
import torchvision
# from torchvision.models import resnet18 as ResNet18
from net.ResNet import ResNet18, Three_Layer_Network

def evaluate(model, val_iter, device):
    correct, total = 0, len(val_iter.dataset)
    for X, y in val_iter:
        X, y = X.to(device), y.to(device)
        y_hat = model(X)
        pred = y_hat.argmax(dim=1)
        correct += torch.eq(pred, y).sum().item()
    return correct / total

def train_with_resnet18(train_iter, val_iter, epochs, lr, device):
    model = Three_Layer_Network().to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr) # Adam优化器
    criterion = nn.CrossEntropyLoss() # 交叉熵损失函数

    # 开始训练
    best_epoch, best_acc = 0, 0
    for epoch in range(epochs):
        for step, (X, y) in enumerate(train_iter):
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            loss = criterion(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 10 == 0:
                print(f'epoch {epoch}, step {step}, loss {loss.item()}')
        if epoch % 1 == 0:
            val_acc = evaluate(model, val_iter, device)
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                torch.save(model.state_dict(), 'best_resnet18.pth')
            print(f'epoch {epoch}, val acc {val_acc}, best acc {best_acc} at epoch {best_epoch}')
    print(f'best acc {best_acc} at epoch {best_epoch}')

def test_with_resnet18(test_iter, device):
    model = ResNet18(pretrained=False)
    model = nn.Sequential(
        *list(model.children())[:-1],
        nn.Flatten(),
        nn.Linear(512, 5)
    ).to(device)
    model.load_state_dict(torch.load('best_resnet18.pth'))
    test_acc = evaluate(model, test_iter, device)
    print(f'test acc {test_acc}')

if __name__ == '__main__':
    from data import Pokemon
    batch_size = 16
    lr = 3e-4
    epochs = 30
    device = torch.device('cuda')
    print("使用设备：", device)
    train_db = Pokemon('data/pokemon', 224, 'train')
    val_db = Pokemon('data/pokemon', 224, 'val')
    test_db = Pokemon('data/pokemon', 224, 'test')
    train_iter = torch.utils.data.DataLoader(train_db, batch_size, shuffle=True, num_workers=4)
    val_iter = torch.utils.data.DataLoader(val_db, batch_size, shuffle=False, num_workers=4)
    test_iter = torch.utils.data.DataLoader(test_db, batch_size, shuffle=False, num_workers=4)

    train_with_resnet18(train_iter, val_iter, epochs, lr, device)
    test_with_resnet18(test_iter, device)
