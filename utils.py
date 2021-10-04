import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from process import get_data
from models import TimmModels
    
def spilt_train_vaild_test(fusai=False):
    df=get_data()
    train_df=df[df['is_train']==1].copy().reset_index(drop=True)
    test=df[df['is_train']==0].copy().reset_index(drop=True)
    if fusai:
        test=get_fusai_test()
    ind_vaild=[]
    for i in range(train_df.category_id.nunique()):
        ind_vaild.extend(train_df[train_df.category_id==i].sample(frac=0.1).index.to_list())
    ind_train=train_df.drop(index=ind_vaild).index.to_list()
    train=train_df.loc[ind_train,:].copy().reset_index(drop=True)
    vaild=train_df.loc[ind_vaild,:].copy().reset_index(drop=True)
    vaild['is_train']=0
        
    train=train.sample(frac=1,random_state=2021).reset_index(drop=True)
    vaild=vaild.sample(frac=1,random_state=2021).reset_index(drop=True)
    return train,vaild,test

def get_fusai_test():
    fusai_test_path='../input/competition-ads-classification-data/fusai_test'
    data=[]
    for imgs_path in [os.path.join(fusai_test_path,_) for _ in os.listdir(fusai_test_path)]:
        image_id=imgs_path.split('/')[-1]
        item={}
        item['is_train']=0
        item['image_id']=image_id
        item['path']=imgs_path
        item['category_id']=0
        data.append(item)
    df=pd.DataFrame(data)
    return df

def vaild_model(model,vaild_dataloader,flag=True):
    image_width,image_height=320
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    s=time.time()
    model.eval()
    correct, total  = 0, 0
    with torch.no_grad():
        for i, (images, labels, orders, image_id) in enumerate(vaild_dataloader):
            images = Variable(images).to(device)
            predict_label = model(images)
            for k,each in enumerate(predict_label):
                # 根据预测结果取值
                predict = np.argmax(each.data.cpu().numpy())
                total += 1
                if predict == orders[k].item():
                    correct += 1
                else:
                    if flag:
                        print('Fail, image_id:%s->%s' % (orders[k].item(), predict))
    if flag:
        print(f'完成。总预测图片数为{total}张，准确率为{int(100 * correct / total)}%，耗时{int(time.time()-s)}s')
    else:
        return correct / total, int(time.time()-s)
    
def predict_model(model,test_dataloader):
    image_width,image_height=320
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predict_list=[]
    model.eval()
    with torch.no_grad():
        for images, labels, orders, image_id in tqdm(test_dataloader):
            predict_label = model(Variable(images.reshape(-1,3,image_width,image_height)).to(device))
            predict = np.argmax(predict_label.data.cpu().numpy())
            predict_list.append({'image_id':image_id[0],'category_id':predict})
    submit=pd.DataFrame(predict_list)
    return submit
    
def get_models():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models_path1='../input/competition-ads-classification-data/models1/'
    models_path2='../input/competition-ads-classification-data/all_train/'
    model_name_list=[]
    model_list=[]
    num=1
    
    model=TimmModels(pretrained=False).to(device)
    model.load_state_dict(torch.load(models_path1+'resnext50_32x4d_0.827808112324493.pkl',map_location=device))
    model_list.append([model,0.827808112324493])
    model_name_list.append(f'resnext50_32x4d{num}')
    num+=1
    
    model=TimmModels(pretrained=False).to(device)
    model.load_state_dict(torch.load(models_path1+'resnext50_32x4d_0.8289781591263651.pkl',map_location=device))
    model_list.append([model,0.8289781591263651])
    model_name_list.append(f'resnext50_32x4d{num}')
    num+=1
    
    model=TimmModels(pretrained=False).to(device)
    model.load_state_dict(torch.load(models_path1+'resnext50_32x4d_0.8297581903276131.pkl',map_location=device))
    model_list.append([model,0.8297581903276131])
    model_name_list.append(f'resnext50_32x4d{num}')
    num+=1
    
    model=TimmModels(pretrained=False).to(device)
    model.load_state_dict(torch.load(models_path1+'resnext50_32x4d_0.8336583463338534.pkl',map_location=device))
    model_list.append([model,0.8336583463338534])
    model_name_list.append(f'resnext50_32x4d{num}')
    num+=1
    
    model=TimmModels(pretrained=False).to(device)
    model.load_state_dict(torch.load(models_path1+'resnext50_32x4d_0.8346333853354134.pkl',map_location=device))
    model_list.append([model,0.8346333853354134])
    model_name_list.append(f'resnext50_32x4d{num}')
    num+=1
    
    model=TimmModels(pretrained=False).to(device)
    model.load_state_dict(torch.load(models_path1+'resnext50_32x4d_0.8354134165366615.pkl',map_location=device))
    model_list.append([model,0.8354134165366615])
    model_name_list.append(f'resnext50_32x4d{num}')
    num+=1
    
    model=TimmModels(pretrained=False).to(device)
    model.load_state_dict(torch.load(models_path1+'resnext50_32x4d_0.8373634945397815.pkl',map_location=device))
    model_list.append([model,0.8373634945397815])
    model_name_list.append(f'resnext50_32x4d{num}')
    num+=1
    
    model=TimmModels(pretrained=False).to(device)
    model.load_state_dict(torch.load(models_path1+'resnext50_32x4d_0.842628705148206.pkl',map_location=device))
    model_list.append([model,0.842628705148206])
    model_name_list.append(f'resnext50_32x4d{num}')
    num+=1
    
    model=TimmModels(pretrained=False).to(device)
    model.load_state_dict(torch.load(models_path1+'resnext50_32x4d_0.8504290171606864.pkl',map_location=device))
    model_list.append([model,0.8504290171606864])
    model_name_list.append(f'resnext50_32x4d{num}')
    num+=1
    
    model=TimmModels(pretrained=False).to(device)
    model.load_state_dict(torch.load(models_path1+'resnext50_32x4d1_0.8504290171606864.pkl',map_location=device))
    model_list.append([model,0.8504290171606864])
    model_name_list.append(f'resnext50_32x4d{num}')
    num+=1
    
    model=TimmModels(pretrained=False).to(device)
    model.load_state_dict(torch.load(models_path1+'resnext50_32x4d_0.8525741029641186.pkl',map_location=device))
    model_list.append([model,0.8525741029641186])
    model_name_list.append(f'resnext50_32x4d{num}')
    num+=1
    
    model=TimmModels(pretrained=False).to(device)
    model.load_state_dict(torch.load(models_path1+'resnext50_32x4d_0.8535491419656787.pkl',map_location=device))
    model_list.append([model,0.8535491419656787])
    model_name_list.append(f'resnext50_32x4d{num}')
    num+=1
    
    model=TimmModels(pretrained=False).to(device)
    model.load_state_dict(torch.load(models_path1+'resnext50_32x4d_0.8547191887675507.pkl',map_location=device))
    model_list.append([model,0.8547191887675507])
    model_name_list.append(f'resnext50_32x4d{num}')
    num+=1
    
    model=TimmModels(pretrained=False).to(device)
    model.load_state_dict(torch.load(models_path1+'resnext50_32x4d_0.8562792511700468.pkl',map_location=device))
    model_list.append([model,0.8562792511700468])
    model_name_list.append(f'resnext50_32x4d{num}')
    num+=1
    
    model=TimmModels(pretrained=False).to(device)
    model.load_state_dict(torch.load(models_path1+'resnext50_32x4d_0.8570592823712948.pkl',map_location=device))
    model_list.append([model,0.8570592823712948])
    model_name_list.append(f'resnext50_32x4d{num}')
    num+=1
    
    model=TimmModels(pretrained=False).to(device)
    model.load_state_dict(torch.load(models_path1+'resnext50_32x4d_0.859009360374415.pkl',map_location=device))
    model_list.append([model,0.859009360374415])
    model_name_list.append(f'resnext50_32x4d{num}')
    num+=1
    
    model=TimmModels(pretrained=False).to(device)
    model.load_state_dict(torch.load(models_path1+'resnext50_32x4d_0.859594383775351.pkl',map_location=device))
    model_list.append([model,0.859594383775351])
    model_name_list.append(f'resnext50_32x4d{num}')
    num+=1
    
    model=TimmModels(pretrained=False).to(device)
    model.load_state_dict(torch.load(models_path1+'resnext50_32x4d_0.8617394695787831.pkl',map_location=device))
    model_list.append([model,0.8617394695787831])
    model_name_list.append(f'resnext50_32x4d{num}')
    num+=1
    
    model=TimmModels(pretrained=False).to(device)
    model.load_state_dict(torch.load(models_path1+'resnext50_32x4d_0.8625195007800313.pkl',map_location=device))
    model_list.append([model,0.8625195007800313])
    model_name_list.append(f'resnext50_32x4d{num}')
    num+=1
    
    model=TimmModels(pretrained=False).to(device)
    model.load_state_dict(torch.load(models_path2+'all_train_epoch17_0.85835.pkl',map_location=device))
    model_list.append([model,0.85835])
    model_name_list.append(f'resnext50_32x4d{num}')
    num+=1
    
    model=TimmModels(pretrained=False).to(device)
    model.load_state_dict(torch.load(models_path2+'all_train_epoch17_0.85897.pkl',map_location=device))
    model_list.append([model,0.85897])
    model_name_list.append(f'resnext50_32x4d{num}')
    num+=1
    
    model=TimmModels(pretrained=False).to(device)
    model.load_state_dict(torch.load(models_path2+'all_train_epoch17_0.85803.pkl',map_location=device))
    model_list.append([model,0.85803])
    model_name_list.append(f'resnext50_32x4d{num}')
    num+=1
    
    num = 1
    
    model = models.resnet34(pretrained=False)
    model.fc = Linear(in_features=512, out_features=137, bias=True)
    model = model.to(device)
    model.load_state_dict(torch.load(models_path1+'resnet34_0.8350234009360374.pkl',map_location=device))
    model_list.append([model,0.8350234009360374])
    model_name_list.append(f'resnet34{num}')
    num+=1

    models_mapping=dict(zip(model_name_list,model_list))
    return models_mapping