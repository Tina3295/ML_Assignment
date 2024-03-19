import os
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from torchvision import models
from torchvision.utils import make_grid
from torchvision import transforms as tsfm
from torchvision.datasets import ImageFolder

import random
import matplotlib.pyplot as plt # 繪製圖表
from tqdm.auto import tqdm # 進度條
from PIL import Image
from pathlib import Path
# from IPython import display

# Set random seed for reproducibility
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.backends.cudnn.deterministic = True


# Set Hyperparameters
data_dir = r'C:\Users\Tina3295\Downloads\plant-seedlings-classification' # dataset's dir you want to unzip
batch_size = 64
epochs = 50
learning_rate = 0.001
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')


# 1. Custom Pytorch Dataset
class Train_data(Dataset):
    def __init__(self, root_dir, transform=None):
        self.data = ImageFolder(root=root_dir, transform=transform)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        return img, label

class Pred_data(Dataset):
    def __init__(self, root_dir, transform=None):
        self.img_paths = list(Path(root_dir).glob('*.png'))
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        img = self.transform(img)
        img = img.unsqueeze(0) # 添加一個額外的維度
        return img


# visualize dataset item for debug
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
# 圖像先被調整為224x224的大小，然後轉換為PyTorch張量
transform = tsfm.Compose([
    tsfm.Resize((224, 224)),
    tsfm.ToTensor(),
])

whole_set = Train_data(
    root_dir=train_dir,
    transform=transform
)

test_set = Pred_data(
    root_dir=test_dir,
    transform=transform
)


# # 一個圖表中顯示5個圖像
# num_images_to_display = 5
# # 創建了一個包含5個subplot的圖表，並指定了圖表的大小為(15, 3)英寸。return了一個包含圖表對象和軸對象的元組
# fig, axs = plt.subplots(1, num_images_to_display, figsize=(15, 3))

# for i, (img, label) in enumerate(whole_set):
#     # 將 PyTorch 張量 img 的維度從 (channel, height, width) 轉換為 (height, width, channel)，這樣才能被 matplotlib 正確地顯示
#     axs[i].imshow(img.permute(1, 2, 0))
#     axs[i].set_title(f'Class: {label}')
#     # 關閉subplot的坐標軸
#     axs[i].axis('off')

#     num_images_to_display -= 1
#     if num_images_to_display == 0:
#         break

# # 調整子圖的間距
# plt.tight_layout()
# plt.show()

# num_images_to_display = 5
# fig, axs = plt.subplots(1, num_images_to_display, figsize=(15, 3))
# for i, img in enumerate(test_set):
#     axs[i].imshow(img[0].permute(1, 2, 0))
#     axs[i].set_title(f'Test img: {i}')
#     axs[i].axis('off')

#     num_images_to_display -= 1
#     if num_images_to_display == 0:
#         break

# plt.tight_layout()
# plt.show()


# 2. Split train, valid set and Create Dataloader:
# 隨機將數據集分割為多個子集。whole_set 數據集被分割為兩個子集，分別佔原始數據集的80%和20%。第一個參數是待分割的數據集，第二個參數是一個包含兩個元素的列表，表示每個子集的比例。返回值是包含分割後子集的 Dataset 物件
train_set, valid_set = random_split(whole_set, [0.8, 0.2])

# train_set 是要被分割的數據集，batch_size 是每個小批次的樣本數量，shuffle=True 表示在每個 epoch 開始時將數據重新洗牌，以增加訓練的隨機性。返回的 train_loader 將在每次迭代時產生一個小批次的訓練數據。
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size)


#3. Create Model
class resnet_50(nn.Module):
    def __init__(self, num_classes = 12):  # num_classes參數指定了模型的輸出類別數量，默認值為12
        super(resnet_50, self).__init__()
        # pytorch built-in models (使用預設的權重的 ResNet-50 初始化模型)
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # set model layers trainable (允許在訓練過程中更新模型的權重參數)
        for param in self.resnet50.parameters():
            param.requires_grad = True

        # redifine/customize last classification layer (重新定義 ResNet-50 模型的最後一個全連接層 (fc)，將其輸出特徵的維度從 2048 調整為 num_classes，以符合輸出類別數量)
        self.resnet50.fc = nn.Linear(2048, num_classes)

    def forward(self, x):  # 定義了模型的前向傳播方法
        x = self.resnet50(x)
        return x
    

# test model for debug
model = resnet_50(num_classes=12).cuda()
# # print(model)
# x = torch.rand(1, 3, 224, 224).cuda()  #(批次大小，圖像通道數（RGB），圖像高度和寬度)
# y = model(x)
# print(x)
# print(y)
    


# 4. Define Train Function(for one epoch):
def train(model, criterion, optimizer, train_loader, epoch, total_epochs, batch_size):
    model.train()
    train_loss, train_acc = [], []

    # 創建進度條
    tqdm_iter = tqdm(train_loader, desc="Epoch: {}/{} ({}%) | Training loss: NaN".format(
    epoch, total_epochs, int(epoch/total_epochs * 100)), leave=False)
    epoch_loss, epoch_acc = 0.0, 0.0
    for batch_idx, (data, label) in enumerate(tqdm_iter):
        data, target = data.cuda(), label.cuda()  # 將輸入數據和標籤移動到CUDA設備上
        output = model(data)  # 對輸入數據進行前向傳播，得到模型的預測結果
        loss = criterion(output, target)
        optimizer.zero_grad()  # 每次迭代前，清空優化器的梯度緩衝區，以便進行下一輪的梯度計算
        loss.backward()  # 透過損失函數計算的梯度反向傳播，計算模型參數的梯度
        optimizer.step()  # 使用優化器來更新模型的參數，以最小化損失函數
        acc = (output.argmax(dim=1) == target).float().mean().item()
        epoch_loss += loss.item()
        epoch_acc += acc

        tqdm_iter.set_description("Epoch: {}/{} ({}%) | Training loss: {:.6f} | Training Acc: {:.6f}".format(
        epoch + 1, total_epochs, int((epoch+1)/total_epochs * 100), round(loss.item(), 6), round(acc, 6)))

    return epoch_loss / len(train_loader), epoch_acc / len(train_loader)


# debug "train" function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# loss, acc = train(
#     model,
#     criterion,
#     optimizer,
#     train_loader,
#     epoch=1,
#     total_epochs=1,
#     batch_size=batch_size
# )

# print(loss, acc)


# 5. Define Valid Function(for one epoch)
def valid(model, criterion, valid_loader, epoch, total_epochs, batch_size):
    model.eval()  # 將模型切換到不計算梯度的模式，以減少記憶體佔用和加快推斷速度。

    tqdm_iter = tqdm(valid_loader, desc="Epoch: {}/{} ({}%) | Valid loss: NaN".format(
    epoch, total_epochs, int(epoch/total_epochs * 100)), leave=False)
    epoch_loss, epoch_acc = 0.0, 0.0
    with torch.no_grad():  # 關閉梯度計算
        for batch_idx, (data, label) in enumerate(tqdm_iter):
            data, target = data.cuda(), label.cuda()
            output = model(data)
            loss = criterion(output, target)
            acc = (output.argmax(dim=1) == target).float().mean().item()
            epoch_loss += loss.item()
            epoch_acc += acc

            tqdm_iter.set_description("Epoch: {}/{} ({}%) | Valid loss: {:.6f} | Valid Acc: {:.6f}".format(
            epoch + 1, total_epochs, int((epoch+1)/total_epochs * 100), round(loss.item(), 6), round(acc, 6)))

    return epoch_loss / len(valid_loader), epoch_acc / len(valid_loader)



# debug "valid" function
# criterion = nn.CrossEntropyLoss()
# loss, acc = valid(
#     model,
#     criterion,
#     valid_loader,
#     epoch=1,
#     total_epochs=1,
#     batch_size=batch_size
# )

# print(loss, acc)



# 6. Plot Learning Curve Function:
def Plot(title, ylabel, epochs, train_loss, valid_loss):  # 繪製訓練和驗證損失隨著訓練週期（epochs）變化的圖表
    plt.figure()
    plt.title(title)
    plt.xlabel('epochs')
    plt.ylabel(ylabel)
    plt.plot(epochs, train_loss)
    plt.plot(epochs, valid_loss)
    plt.legend(['train', 'valid'], loc='upper left')  # 圖例位置


# # debug "Plot" function
# debug_epochs = [1, 2, 3, 4, 5]
# debug_train_loss = [0.1, 0.08, 0.06, 0.05, 0.04]
# debug_valid_loss = [0.2, 0.15, 0.12, 0.1, 0.09]

# Plot("Training and Validation Loss", 'Loss', debug_epochs, debug_train_loss, debug_valid_loss)

# plt.show()
    


# 7. Predict Function:
def predict(loader, model):
    model.eval()  # 設置模型處於評估模式
    preds = []
    for data in tqdm(loader):
        pred = model(data.cuda())
        cls = torch.argmax(pred, dim=1)  # 對於每個預測結果，找到具有最高概率的類別
        preds.append(cls)

    return preds

# Visualize Predict result
def view_pred_result(preds, num_images_to_display=5):
    labels = ['Black-grass', 'Charlock' , 'Cleavers' , 'Common Chickweed' , 'Common wheat' , 'Fat Hen' , 'Loose Silky-bent' , 'Maize' , 'Scentless Mayweed' , 'Shepherds Purse', 'Small-flowered Cranesbill' , 'Sugar beet']
    # 創建一個包含多個子圖的圖表，每個子圖顯示一張圖像及其對應的預測類別。
    fig, axs = plt.subplots(1, num_images_to_display, figsize=(15, 3))
    for i, img in enumerate(test_set):
        axs[i].imshow(img[0].permute(1, 2, 0))  # 在第i個子圖上顯示圖像，需要將圖像的尺寸從（channel, height, width）轉換為（height, width, channel），以便matplotlib正確顯示
        axs[i].set_title(labels[preds[i].item()])
        axs[i].axis('off') # 關閉子圖的坐標軸

        num_images_to_display -= 1
        if num_images_to_display == 0:
            break

    plt.tight_layout()
    plt.show()

# debug "Predict" function & "View_Predict_result" function
test_dir = os.path.join(data_dir, 'test')
# transform = tsfm.Compose([
#     tsfm.Resize((224, 224)),
#     tsfm.ToTensor(),
# ])
# test_set = Pred_data(
#     root_dir=test_dir,
#     transform=transform
# )
# model = resnet_50(num_classes=12).cuda()

# preds = predict(test_set, model)
# view_pred_result(preds)



# # 8. Main Function(training pipeline):
# def main():
#     # initial transform
#     transform = tsfm.Compose([
#         tsfm.Resize((224, 224)),
#         tsfm.ToTensor(),
#     ])

#     # initial dataset
#     whole_set = Train_data(
#         root_dir=train_dir,
#         transform=transform
#     )

#     test_set = Pred_data(
#         root_dir=test_dir,
#         transform=transform
#     )

#     # split train valid and initial dataloader
#     train_set_size = int(len(whole_set) * 0.8)
#     valid_set_size = len(whole_set) - train_set_size
#     train_set, valid_set = random_split(whole_set, [train_set_size, valid_set_size])

#     train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
#     valid_loader = DataLoader(valid_set, batch_size=batch_size)

#     # initial model
#     model = resnet_50(num_classes=12).cuda()

#     # initial loss_function and optimizer
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#     # initial plot values
#     train_loss, train_acc = [], []
#     valid_loss, valid_acc = [], []
#     epoch_list = []

#     # repeat train and valid epochs times
#     print(epochs)
#     for epoch in range(epochs):
#       epoch_list.append(epoch + 1)

#       loss, acc = train(
#           model,
#           criterion,
#           optimizer,
#           train_loader,
#           epoch=epoch,
#           total_epochs=epochs,
#           batch_size=batch_size
#       )
#       train_loss.append(loss)
#       train_acc.append(acc)
#       print(f'Avg train Loss: {loss}, Avg train acc: {acc}')

#       loss, acc = valid(
#           model,
#           criterion,
#           valid_loader,
#           epoch=epoch,
#           total_epochs=epochs,
#           batch_size=batch_size
#       )
#       valid_loss.append(loss)
#       valid_acc.append(acc)
#       print(f'Avg valid Loss: {loss}, Avg valid acc: {acc}')

#     Plot("Loss Curve", 'Loss', epoch_list, train_loss, valid_loss)
#     Plot("Accuarcy Curve", 'Acc', epoch_list, train_acc, valid_acc)

#     preds = predict(test_set, model)
#     view_pred_result(preds)

# main()




# 9. Addition: Customize your own model
class VGG16(nn.Module):
    def __init__(self, num_classes=12):
        super(VGG16, self).__init__()
        # input layer
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())

        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        #  classifier
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7*7*512, 4096),
            nn.ReLU()
        )

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU()
        )

        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
    


# # Test model to debug
# x = torch.rand(1, 3, 224, 224)
# model = VGG16(num_classes=12)
# y = model(x)
# print(y)
    

def main():
    # initial transform
    transform = tsfm.Compose([
        tsfm.Resize((224, 224)),
        tsfm.ToTensor(),
    ])

    # initial dataset
    whole_set = Train_data(
        root_dir=train_dir,
        transform=transform
    )

    test_set = Pred_data(
        root_dir=test_dir,
        transform=transform
    )

    # split train valid and initial dataloader
    train_set_size = int(len(whole_set) * 0.8)
    valid_set_size = len(whole_set) - train_set_size
    train_set, valid_set = random_split(whole_set, [train_set_size, valid_set_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size)

    # initial model
    model = VGG16(num_classes=12).cuda()

    # initial loss_function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # initial plot values
    train_loss, train_acc = [], []
    valid_loss, valid_acc = [], []
    epoch_list = []

    # repeat train and valid epochs times
    print(epochs)
    for epoch in range(epochs):
      epoch_list.append(epoch + 1)

      loss, acc = train(
          model,
          criterion,
          optimizer,
          train_loader,
          epoch=epoch,
          total_epochs=epochs,
          batch_size=batch_size
      )
      train_loss.append(loss)
      train_acc.append(acc)
      print(f'Avg train Loss: {loss}, Avg train acc: {acc}')

      loss, acc = valid(
          model,
          criterion,
          valid_loader,
          epoch=epoch,
          total_epochs=epochs,
          batch_size=batch_size
      )
      valid_loss.append(loss)
      valid_acc.append(acc)
      print(f'Avg valid Loss: {loss}, Avg valid acc: {acc}')

    Plot("Loss Curve", 'Loss', epoch_list, train_loss, valid_loss)
    Plot("Accuarcy Curve", 'Acc', epoch_list, train_acc, valid_acc)

    preds = predict(test_set, model)
    view_pred_result(preds)

main()