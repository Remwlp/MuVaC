import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torchmetrics
import model  
from sklearn.model_selection import train_test_split
import pickle
import torch.nn.functional as F
import os    
from transformers import BartTokenizerFast,AdamW
from torch.utils.data import DataLoader, TensorDataset
from data_util import MyDataset, load_data
import random
import gc
import numpy as np
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BartTokenizerFast.from_pretrained('bart-base')
tokenizer.add_tokens(['[CONTEXT]', '[TARGET]'], special_tokens=True)

# 加载配置
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config


def kl_div_gaussian(p_mean, p_var, q_mean, q_var):
    var_p_div_q = p_var / q_var.clamp(min=1e-6)
    kl_div = (-torch.log(var_p_div_q) + var_p_div_q - 1 + (p_mean - q_mean) ** 2 / q_var).sum(-1).mean() / 2
    return kl_div

def prepare_for_training(model, 
                         base_learning_rate: float,
                         new_learning_rate: float,
                         weight_decay: float):
    base_params_list = []
    new_params_list = []
    for name, param in model.named_parameters():
        if  "My" in name or "exp_feature" in name:
            new_params_list.append(param)
        else:
            base_params_list.append(param)
            
    optimizer = AdamW(
        [
            {'params': base_params_list,'lr': base_learning_rate, 'weight_decay': weight_decay},
            {'params': new_params_list,'lr': new_learning_rate, 'weight_decay': weight_decay}            
        ]
    )
    
    return optimizer




# 训练和评估函数，现在接收完整的config对象
def train_and_evaluate(config):
    num_epochs = config['num_epochs']
    base_learning_rate = config['BASE_LEARNING_RATE']
    new_learning_rate = config['NEW_LEARNING_RATE']
    weight_decay = config['WEIGHT_DECAY']
    batch_size = config['batch_size']
    model_params = config['model_params']
    data_path = config['dataset_path']
    output_dir = config['output_dir']
    max_len = config['exp_max_len']


    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    random.seed(config['seed'])
    os.environ["PYTHONHASHSEED"] = str(config['seed'])
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False  
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = False



    train_loader, test_loader,val_loader = load_data(data_path, batch_size, max_len, tokenizer,config['seed'])

    my_model = model.MyModel(
        tokenizer = tokenizer
    )
    my_model.to(device)


    criterion_class = nn.CrossEntropyLoss()
    optimizer = prepare_for_training(my_model, base_learning_rate,  new_learning_rate, weight_decay)

    accuracy = torchmetrics.Accuracy(num_classes=2, average='weighted', task='binary').to(device)
    precision = torchmetrics.Precision(num_classes=2, average='weighted', task='binary').to(device)
    recall = torchmetrics.Recall(num_classes=2, average='weighted', task='binary').to(device)
    f1_score = torchmetrics.F1Score(num_classes=2, average='weighted', task='binary').to(device)

    best_loss = torch.inf
    best_acc = 0.0
    for epoch in range(num_epochs):
        my_model.train()
        accuracy.reset()
        precision.reset()
        recall.reset()
        f1_score.reset()    

        epoch_loss = 0.0
        num_batches = 0
        train_prog_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for input_id, attention_mask, audio_feature, face_feature, posture_feature, video_feature, exp, labels in train_prog_bar:
            input_id, attention_mask, audio_feature, face_feature, posture_feature, video_feature, exp, labels = (
                input_id.to(device),
                attention_mask.to(device),
                audio_feature.to(device),
                face_feature.to(device),
                posture_feature.to(device),
                video_feature.to(device),
                exp.to(device),
                labels.to(device)
            )
            optimizer.zero_grad()
            class_output, exp_loss, q_mean, q_var, p_mean, p_var, exp_id = my_model(input_id, attention_mask, audio_feature, face_feature, posture_feature, video_feature, exp, labels)

            ans_loss = criterion_class(class_output, labels)
            preds = class_output.argmax(-1)


            kl_div = kl_div_gaussian(q_mean, q_var, p_mean, p_var)
            loss = ans_loss + kl_div + exp_loss 

            epoch_loss += loss.item()
            num_batches += 1
            avg_epoch_loss = epoch_loss / num_batches


            loss.backward()
            optimizer.step()

            accuracy.update(preds, labels)
            precision.update(preds, labels)
            recall.update(preds, labels)
            f1_score.update(preds, labels)
            train_prog_bar.set_postfix(loss=avg_epoch_loss, acc=accuracy.compute().item(), 
                                        precision=precision.compute().item(), 
                                        recall=recall.compute().item(), f1=f1_score.compute().item())

        
                
        my_model.eval()
        accuracy.reset()
        precision.reset()
        recall.reset()
        f1_score.reset()    
        val_loss = 0.0
        val_batches = 0
        val_prog_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Valid]")
        with torch.no_grad():
            for input_id, attention_mask, audio_feature, face_feature, posture_feature, video_feature, exp, labels in val_prog_bar:
                input_id, attention_mask, audio_feature, face_feature, posture_feature, video_feature, exp, labels = (
                input_id.to(device),
                attention_mask.to(device),
                audio_feature.to(device),
                face_feature.to(device),
                posture_feature.to(device),
                video_feature.to(device),
                exp.to(device),
                labels.to(device)
                )

                class_output, exp_loss, q_mean, q_var, p_mean, p_var, exp_id = my_model(input_id, attention_mask, audio_feature, face_feature, posture_feature, video_feature,exp)
                preds = class_output.argmax(-1)

                cls_labels = F.one_hot(labels.to(torch.int64), num_classes=2).float()  
                loss_class = criterion_class(class_output, cls_labels)

                val_loss += loss_class.item()
                val_batches += 1
                avg_loss = val_loss/val_batches
                accuracy.update(preds, labels)
                precision.update(preds, labels)
                recall.update(preds, labels)
                f1_score.update(preds, labels)                
                val_prog_bar.set_postfix(loss = avg_loss,acc=accuracy.compute().item(), 
                                         precision=precision.compute().item(), 
                                         recall=recall.compute().item(), f1=f1_score.compute().item())

        
        my_model.eval()
        accuracy.reset()
        precision.reset()
        recall.reset()
        f1_score.reset()  
        test_loss = 0.0  
        test_batches = 0
        test_prog_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Test]")
        with torch.no_grad():
            for input_id, attention_mask, audio_feature, face_feature, posture_feature, video_feature, exp, labels in test_prog_bar:
                input_id, attention_mask, audio_feature, face_feature, posture_feature, video_feature, exp, labels = (
                input_id.to(device),
                attention_mask.to(device),
                audio_feature.to(device),
                face_feature.to(device),
                posture_feature.to(device),
                video_feature.to(device),
                exp.to(device),
                labels.to(device)
                )

                class_output, exp_loss, q_mean, q_var, p_mean, p_var, exp_id = my_model(input_id, attention_mask, audio_feature, face_feature, posture_feature, video_feature,exp)
                preds = class_output.argmax(-1)


                cls_labels = F.one_hot(labels.to(torch.int64), num_classes=2).float() 
                loss_class = criterion_class(class_output, cls_labels)
                test_loss += loss_class.item()
                test_batches += 1
                avg_loss = test_loss/test_batches
                accuracy.update(preds, labels)
                precision.update(preds, labels)
                recall.update(preds, labels)
                f1_score.update(preds, labels)                
                test_prog_bar.set_postfix(loss = avg_loss, acc=accuracy.compute().item(), 
                                         precision=precision.compute().item(), 
                                         recall=recall.compute().item(), f1=f1_score.compute().item())
    return 
# 主逻辑
if __name__ == "__main__":
    config = load_config('config.json')
    train_and_evaluate(config)