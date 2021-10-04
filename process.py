import os
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

def get_data():
    df_path='../input/competition-ads-classification-data/dataframe/data.csv'
    if os.path.exists(df_path):
        df=pd.read_csv(df_path)
    else:
        base_path='../input/competition-ads-classification-data/data/'
        train_path=os.path.join(base_path,'训练集')
        test_path=os.path.join(base_path,'初赛_测试集')
        all_labels=list(range(137))
        data=[]
        for i in tqdm(range(len(all_labels))):
            vector=np.zeros(len(all_labels),dtype=float)
            vector[i]=1.0
            path=os.path.join(train_path,str(i))
            imgs_paths=[os.path.join(path,_) for _ in os.listdir(path)]
            for imgs_path in imgs_paths:
                width,height=Image.open(imgs_path).size
                image_id=imgs_path.split('/')[-1]
                item={}
                item['is_train']=1
                item['image_id']=image_id
                item['path']=imgs_path
                item['width']=width
                item['height']=height
                item['vector']=vector
                item['category_id']=i
                data.append(item)

        for imgs_path in tqdm([os.path.join(test_path,_) for _ in os.listdir(test_path)]):
            width,height=Image.open(imgs_path).size
            image_id=imgs_path.split('/')[-1]
            item={}
            item['is_train']=0
            item['image_id']=image_id
            item['path']=imgs_path
            item['width']=width
            item['height']=height
            data.append(item)
        df=pd.DataFrame(data)
        df.to_csv(df_path,index=False)
    df=df.assign(category_id=df.category_id.fillna(0).astype(int))
    return df
    
def print_img_mean_std():
    img_filenames = get_data().path.values.tolist()
    m_list, s_list = [], []
    img_h, img_w = 320, 320
    for img_filename in tqdm(img_filenames):
        img=Image.open(img_filename).convert('RGBA')
        img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGBA2BGR)
        img = cv2.resize(img, (img_h, img_w))
        img = img / 255.0
        m, s = cv2.meanStdDev(img)
        m_list.append(m.reshape((3,)))
        s_list.append(s.reshape((3,)))
    m_array = np.array(m_list)
    s_array = np.array(s_list)
    m = m_array.mean(axis=0, keepdims=True)
    s = s_array.mean(axis=0, keepdims=True)
    print('均值：',m[0][::-1])
    print('标准差：',s[0][::-1])