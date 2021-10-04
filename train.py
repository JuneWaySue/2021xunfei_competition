from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import numpy as np
import pandas as pd
import time
import os
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable
from torch.optim import SGD,lr_scheduler

from utils import spilt_train_vaild_test,vaild_model
from dataset import ImageDataSet2
from models import TimmModels


# 训练参数
epochs = 17
batch_size = 50
device = 'cuda' if torch.cuda.is_available() else 'cpu'
verbose=100
image_width=320
image_height=320

# 320
norm_mean = (0.63790344, 0.56811579, 0.5704457)
norm_std = (0.24307405, 0.2520139, 0.25256122)

test_transform = transforms.Compose([
            transforms.Resize((image_width,image_height)),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean,norm_std)
        ])

models_path1='../input/competition-ads-classification-data/models1/'
if not os.path.exists(models_path1):
    os.makedirs(models_path1)

models_path2='../input/competition-ads-classification-data/all_train/'
if not os.path.exists(models_path2):
    os.makedirs(models_path2)

def train_first(rate):
    train,vaild,test=spilt_train_vaild_test(fusai=True)

    train_dataloader=DataLoader(ImageDataSet2(train,test_transform), batch_size=batch_size, shuffle=True, num_workers=32)
    vaild_dataloader=DataLoader(ImageDataSet2(vaild,test_transform), batch_size=batch_size, shuffle=True, num_workers=32)
    test_dataloader=DataLoader(ImageDataSet2(test,test_transform), batch_size=1, shuffle=False, num_workers=32)

    model=TimmModels(pretrained=True).to(device)

    # 损失函数
    criterion = CrossEntropyLoss().to(device)
    # SGD算法
    optimizer = SGD(model.parameters(), lr=rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=len(train_dataloader) / 10, gamma=0.95)
    best_acc = 0
    best_epoch = 0
    print(f'Train start...')
    print(f'train_dataloader iters is {len(train_dataloader)}')
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total=0
        correct=0
        s=time.time() 
        
        for i, (images, labels, orders, image_id) in enumerate(train_dataloader):
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)
            predict_labels = model(images)
            prediction = predict_labels.max(1, keepdim=True)[1]

            total += len(orders)
            correct += torch.eq(prediction.view(-1).cpu(),orders).sum().item()

            loss = criterion(predict_labels, labels)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if (i+1) % verbose == 0:
                print("epoch:{0}, step:{1}, loss:{2:.5f}, train's accuracy:{3:.2%}, time:{4} s".format(epoch+1, i+1, total_loss / verbose, correct / total, int(time.time()-s)))
                total_loss = 0
                total=0
                correct=0
                s=time.time()
        # 验证集得分
        acc,time_cost = vaild_model(model,vaild_dataloader,flag=False)
        print("epoch:{0}, vaild's accuracy:{1:.2%}, time:{2} s".format(epoch+1,acc,time_cost))
        print('-'*50)
        if acc > best_acc:
            # 保存训练结果
            torch.save(model.state_dict(), models_path1+f'result{epoch+1}_{acc}.pkl')
            best_acc=acc
            best_epoch=epoch+1
    print(f'Train done!!!  best_epoch:{best_epoch}  best_acc:{best_acc}')


def train_all(rate):
    train,vaild,test=spilt_train_vaild_test(fusai=True)
    vaild['is_train']=1
    train=pd.concat([train,vaild],ignore_index=True)

    train_dataloader=DataLoader(ImageDataSet2(train,test_transform), batch_size=batch_size, shuffle=True, num_workers=32)
    test_dataloader=DataLoader(ImageDataSet2(test,test_transform), batch_size=1, shuffle=False, num_workers=32)

    model=TimmModels(pretrained=True).to(device)

    # 损失函数
    criterion = CrossEntropyLoss().to(device)
    optimizer = SGD(model.parameters(), lr=rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=len(train_dataloader) / 10, gamma=0.95)
    print(f'Train start...')
    print(f'train_dataloader iters is {len(train_dataloader)}')
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total=0
        correct=0
        s=time.time() 
        
        for i, (images, labels, orders, image_id) in enumerate(train_dataloader):
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)
            predict_labels = model(images)
            prediction = predict_labels.max(1, keepdim=True)[1]

            total += len(orders)
            correct += torch.eq(prediction.view(-1).cpu(),orders).sum().item()

            loss = criterion(predict_labels, labels)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
        acc=correct / total 
        print("epoch:{0}, loss:{1:.5f}, train's accuracy:{2:.2%}, time:{3} s".format(epoch+1, total_loss / i, acc, int(time.time()-s)))
        # 保存训练结果
        torch.save(model.state_dict(), models_path2+f'all_train_epoch{epoch+1}_{rate}.pkl')
    print(f'Train done!!!')


if __name__ == '__main__':
    for rate in np.arange(0.01,0.11,0.01).round(2).tolist()+np.arange(0.15,0.65,0.05).round(2).tolist():
        train_first(rate)

    for rate in [0.09,0.08,0.05]:
        train_all(rate)