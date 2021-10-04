import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import pandas as pd
import time
from tqdm import tqdm

from utils import spilt_train_vaild_test,get_models
from dataset import ImageDataSet


device = 'cuda' if torch.cuda.is_available() else 'cpu'
image_width=320
image_height=320
 
norm_mean = (0.63790344, 0.56811579, 0.5704457)
norm_std = (0.24307405, 0.2520139, 0.25256122)
train,vaild,test=spilt_train_vaild_test(contains_chusai_test=True,fusai=True)

test_transform = transforms.Compose([
            transforms.Resize((image_width,image_height)),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean,norm_std)
        ])

test_dataloader=DataLoader(ImageDataSet(test,test_transform), batch_size=1, shuffle=False, num_workers=32)
vaild_dataloader=DataLoader(ImageDataSet(vaild,test_transform), batch_size=1, shuffle=False, num_workers=32)

models_mapping=get_models()
print('模型数量：',len(models_mapping))
print('模型如下：','、'.join(list(models_mapping.keys())))
time.sleep(0.9)

with torch.no_grad():
    result_all=[]
    for images, labels, orders, image_id in tqdm(test_dataloader):
        pred_array_all=np.zeros(137,dtype=np.float32)
        total_acc_all = 0
        for name,(model,acc) in models_mapping.items():
            model.eval()
            predict_label = model(Variable(images.reshape(-1,3,image_width,image_height)).to(device))
            predict = predict_label.data.cpu().numpy().reshape(-1) * acc
            pred_array_all += predict
            total_acc_all += acc
        pred_all = np.argmax(pred_array_all / total_acc_all)
        result_all.append({'image_id':image_id[0],'category_id':pred_all})

submit=pd.DataFrame(result_all)
submit.to_csv('submit.csv',index=False)