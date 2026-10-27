import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split

DATA_DIR="Data/source"
OUTPUT_DIR="Data/preprocessed"
IMAGE_SIZE=224
SEED=42
CLASS_NAMES=["Non-Stone","Stone"]

class KidneyStoneDataset(Dataset):
    def __init__(self,records,transform=None):
        self.records=records
        self.transform=transform
    def __len__(self):
        return len(self.records)
    def __getitem__(self,index):
        path,label=self.records[index]
        image=Image.open(path).convert("L")
        image=np.asarray(image,dtype=np.float32)
        mean=image.mean()
        std=image.std()
        image=(image-mean)/(std+1e-8)
        image=np.clip(image,-5,5)
        image=(image-image.min())/(image.max()-image.min()+1e-8)
        image=Image.fromarray((image*255).astype(np.uint8)).convert("RGB")
        if self.transform:
            image=self.transform(image)
        return image,torch.tensor(label,dtype=torch.long)

def get_transforms():
    return transforms.Compose([transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),transforms.ToTensor(),transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])

def collect_images(data_dir):
    records=[]
    extensions=(".png",".jpg",".jpeg",".bmp",".tif",".tiff")
    for label,class_name in enumerate(CLASS_NAMES):
        class_dir=os.path.join(data_dir,class_name)
        if not os.path.isdir(class_dir):
            continue
        for root,_,files in os.walk(class_dir):
            for filename in sorted(files):
                if filename.lower().endswith(extensions):
                    records.append((os.path.join(root,filename),label))
    if not records:
        raise FileNotFoundError(f"No class-folder images found in {data_dir}. Expected folders: {CLASS_NAMES}")
    return records

def preprocess_dataset(data_dir=DATA_DIR,output_dir=OUTPUT_DIR):
    os.makedirs(output_dir,exist_ok=True)
    records=collect_images(data_dir)
    labels=[label for _,label in records]
    train_records,test_records=train_test_split(records,test_size=0.30,random_state=SEED,stratify=labels)
    transform=get_transforms()
    train_data=[{"image":path,"label":label} for path,label in train_records]
    test_data=[{"image":path,"label":label} for path,label in test_records]
    metadata={"class_names":CLASS_NAMES,"image_size":IMAGE_SIZE,"seed":SEED,"split_ratio":"70:30","train_count":len(train_data),"test_count":len(test_data),"train":train_data,"test":test_data}
    with open(os.path.join(output_dir,"metadata.json"),"w") as file:
        json.dump(metadata,file,indent=2)
    print(f"Total images: {len(records)}")
    print(f"Training images: {len(train_data)}")
    print(f"Testing images: {len(test_data)}")
    print(f"Metadata saved to {os.path.join(output_dir,'metadata.json')}")
    return train_data,test_data,transform

if __name__=="__main__":
    preprocess_dataset()