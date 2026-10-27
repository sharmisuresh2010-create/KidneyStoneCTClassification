import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Subset
from sklearn.model_selection import train_test_split
from preprocessing import KidneyStoneDataset,collect_images,get_transforms,DATA_DIR,OUTPUT_DIR,CLASS_NAMES
from modelling import build_model,reset_random

SEED=42
BATCH_SIZE=32
EPOCHS=50
LEARNING_RATE=1e-4
WEIGHT_DECAY=1e-4
PATIENCE=10
NUM_WORKERS=4
MODEL_DIR="model"
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model():
    reset_random(SEED)
    os.makedirs(MODEL_DIR,exist_ok=True)
    records=collect_images(DATA_DIR)
    labels=[label for _,label in records]
    train_records,test_records=train_test_split(records,test_size=0.30,random_state=SEED,stratify=labels)
    transform=get_transforms()
    train_dataset=KidneyStoneDataset(train_records,transform)
    test_dataset=KidneyStoneDataset(test_records,transform)
    generator=torch.Generator().manual_seed(SEED)
    train_loader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS,pin_memory=torch.cuda.is_available(),generator=generator)
    test_loader=DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS,pin_memory=torch.cuda.is_available())
    model=build_model(num_classes=len(CLASS_NAMES),pretrained=True).to(DEVICE)
    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS)
    best_loss=float("inf")
    patience_counter=0
    history={"train_loss":[],"test_loss":[],"train_accuracy":[],"test_accuracy":[],"learning_rate":[]}
    best_path=os.path.join(MODEL_DIR,"best_model.pth")
    for epoch in range(EPOCHS):
        model.train()
        train_loss=0.0
        train_correct=0
        train_total=0
        for images,labels_batch in train_loader:
            images=images.to(DEVICE,non_blocking=True)
            labels_batch=labels_batch.to(DEVICE,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs=model(images)
            loss=criterion(outputs,labels_batch)
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()*images.size(0)
            train_correct+=(outputs.argmax(1)==labels_batch).sum().item()
            train_total+=labels_batch.size(0)
        model.eval()
        test_loss=0.0
        test_correct=0
        test_total=0
        with torch.no_grad():
            for images,labels_batch in test_loader:
                images=images.to(DEVICE,non_blocking=True)
                labels_batch=labels_batch.to(DEVICE,non_blocking=True)
                outputs=model(images)
                loss=criterion(outputs,labels_batch)
                test_loss+=loss.item()*images.size(0)
                test_correct+=(outputs.argmax(1)==labels_batch).sum().item()
                test_total+=labels_batch.size(0)
        train_loss/=max(train_total,1)
        test_loss/=max(test_total,1)
        train_accuracy=train_correct/max(train_total,1)
        test_accuracy=test_correct/max(test_total,1)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_accuracy"].append(train_accuracy)
        history["test_accuracy"].append(test_accuracy)
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])
        scheduler.step()
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Train Acc: {train_accuracy:.4f} | Test Acc: {test_accuracy:.4f}")
        if test_loss<best_loss:
            best_loss=test_loss
            patience_counter=0
            torch.save({"epoch":epoch+1,"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),"best_test_loss":best_loss,"class_names":CLASS_NAMES,"seed":SEED},best_path)
        else:
            patience_counter+=1
        with open(os.path.join(MODEL_DIR,"history.json"),"w") as file:
            json.dump(history,file,indent=2)
        if patience_counter>=PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break
    with open(os.path.join(MODEL_DIR,"training_config.json"),"w") as file:
        json.dump({"seed":SEED,"batch_size":BATCH_SIZE,"epochs":EPOCHS,"learning_rate":LEARNING_RATE,"weight_decay":WEIGHT_DECAY,"patience":PATIENCE,"split_ratio":"70:30","best_test_loss":best_loss},file,indent=2)
    print(f"Best model saved to {best_path}")
    print(f"Training history saved to {os.path.join(MODEL_DIR,'history.json')}")

if __name__=="__main__":
    train_model()