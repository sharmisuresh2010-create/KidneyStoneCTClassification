import os
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef,roc_auc_score,average_precision_score,confusion_matrix
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from preprocessing import KidneyStoneDataset,collect_images,get_transforms,DATA_DIR,CLASS_NAMES
from modelling import build_model

SEED=42
BATCH_SIZE=32
MODEL_DIR="model"
RESULTS_DIR="results"
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_metrics(y_true,y_pred,y_prob):
    metrics={"accuracy":accuracy_score(y_true,y_pred),"precision":precision_score(y_true,y_pred,average="binary",zero_division=0),"recall":recall_score(y_true,y_pred,average="binary",zero_division=0),"f1_score":f1_score(y_true,y_pred,average="binary",zero_division=0),"mcc":matthews_corrcoef(y_true,y_pred)}
    if len(np.unique(y_true))==2:
        metrics["roc_auc"]=roc_auc_score(y_true,y_prob[:,1])
        metrics["average_precision"]=average_precision_score(y_true,y_prob[:,1])
        metrics["brier_score"]=brier_score_loss(y_true,y_prob[:,1])
        fraction,mean_probability=calibration_curve(y_true,y_prob[:,1],n_bins=10,strategy="uniform")
        metrics["ece"]=float(np.average(np.abs(fraction-mean_probability),weights=np.histogram(y_prob[:,1],bins=np.linspace(0,1,11))[0][:len(fraction)])) if len(fraction) else None
    else:
        metrics.update({"roc_auc":None,"average_precision":None,"brier_score":None,"ece":None})
    return metrics

def evaluate_split(model,loader,criterion):
    model.eval()
    losses=[]
    y_true=[]
    y_pred=[]
    y_prob=[]
    with torch.no_grad():
        for images,labels in loader:
            images=images.to(DEVICE,non_blocking=True)
            labels=labels.to(DEVICE,non_blocking=True)
            outputs=model(images)
            losses.append(criterion(outputs,labels).item()*images.size(0))
            probabilities=torch.softmax(outputs,dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(probabilities.argmax(1).cpu().numpy())
            y_prob.extend(probabilities.cpu().numpy())
    y_true=np.asarray(y_true)
    y_pred=np.asarray(y_pred)
    y_prob=np.asarray(y_prob)
    return float(sum(losses)/max(len(y_true),1)),y_true,y_pred,y_prob,calculate_metrics(y_true,y_pred,y_prob)

def save_loss_curves(history):
    epochs=range(1,len(history["train_loss"])+1)
    plt.figure(figsize=(8,5))
    plt.plot(epochs,history["train_loss"],label="Train Loss")
    plt.plot(epochs,history["test_loss"],label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Testing Loss")
    plt.legend()
    plt.grid(True,alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR,"loss_curve.png"),dpi=300)
    plt.close()
    plt.figure(figsize=(8,5))
    plt.plot(epochs,history["train_accuracy"],label="Train Accuracy")
    plt.plot(epochs,history["test_accuracy"],label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Testing Accuracy")
    plt.legend()
    plt.grid(True,alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR,"accuracy_curve.png"),dpi=300)
    plt.close()

def save_confusion_matrix(y_true,y_pred,split_name):
    matrix=confusion_matrix(y_true,y_pred,labels=[0,1])
    plt.figure(figsize=(6,5))
    plt.imshow(matrix,interpolation="nearest")
    plt.title(f"{split_name} Confusion Matrix")
    plt.colorbar()
    ticks=np.arange(len(CLASS_NAMES))
    plt.xticks(ticks,CLASS_NAMES,rotation=30)
    plt.yticks(ticks,CLASS_NAMES)
    threshold=matrix.max()/2 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j,i,str(matrix[i,j]),ha="center",va="center",color="white" if matrix[i,j]>threshold else "black")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR,f"{split_name.lower()}_confusion_matrix.png"),dpi=300)
    plt.close()

def save_roc_pr_curves(results):
    plt.figure(figsize=(8,5))
    for name,data in results.items():
        y_true,y_prob=data["y_true"],data["y_prob"]
        if len(np.unique(y_true))==2:
            from sklearn.metrics import roc_curve
            fpr,tpr,_=roc_curve(y_true,y_prob[:,1])
            plt.plot(fpr,tpr,label=f"{name} AUC={roc_auc_score(y_true,y_prob[:,1]):.4f}")
    plt.plot([0,1],[0,1],"--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.grid(True,alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR,"roc_curves.png"),dpi=300)
    plt.close()
    plt.figure(figsize=(8,5))
    for name,data in results.items():
        y_true,y_prob=data["y_true"],data["y_prob"]
        if len(np.unique(y_true))==2:
            from sklearn.metrics import precision_recall_curve
            precision,recall,_=precision_recall_curve(y_true,y_prob[:,1])
            plt.plot(recall,precision,label=f"{name} AP={average_precision_score(y_true,y_prob[:,1]):.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.grid(True,alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR,"pr_curves.png"),dpi=300)
    plt.close()

def main():
    os.makedirs(RESULTS_DIR,exist_ok=True)
    checkpoint_path=os.path.join(MODEL_DIR,"best_model.pth")
    history_path=os.path.join(MODEL_DIR,"history.json")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    records=collect_images(DATA_DIR)
    from sklearn.model_selection import train_test_split
    labels=[label for _,label in records]
    train_records,test_records=train_test_split(records,test_size=0.30,random_state=SEED,stratify=labels)
    transform=get_transforms()
    train_loader=DataLoader(KidneyStoneDataset(train_records,transform),batch_size=BATCH_SIZE,shuffle=False,num_workers=4,pin_memory=torch.cuda.is_available())
    test_loader=DataLoader(KidneyStoneDataset(test_records,transform),batch_size=BATCH_SIZE,shuffle=False,num_workers=4,pin_memory=torch.cuda.is_available())
    model=build_model(num_classes=len(CLASS_NAMES),pretrained=False).to(DEVICE)
    checkpoint=torch.load(checkpoint_path,map_location=DEVICE,weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion=nn.CrossEntropyLoss()
    results={}
    summary={}
    for name,loader in [("Train",train_loader),("Test",test_loader)]:
        loss,y_true,y_pred,y_prob,metrics=evaluate_split(model,loader,criterion)
        metrics["loss"]=loss
        summary[name]=metrics
        results[name]={"y_true":y_true,"y_prob":y_prob}
        save_confusion_matrix(y_true,y_pred,name)
        print(f"\n{name} Metrics")
        for metric,value in metrics.items():
            print(f"{metric}: {value:.4f}" if isinstance(value,(float,np.floating)) else f"{metric}: {value}")
    with open(os.path.join(RESULTS_DIR,"evaluation_metrics.json"),"w") as file:
        json.dump(summary,file,indent=2)
    if os.path.isfile(history_path):
        with open(history_path,"r") as file:
            history=json.load(file)
        save_loss_curves(history)
    save_roc_pr_curves(results)
    print(f"\nEvaluation outputs saved in: {RESULTS_DIR}")

if __name__=="__main__":
    main()