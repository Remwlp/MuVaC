import torch
from torch.utils.data import Dataset
import json
import os
import pickle
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import pandas as pd
import re

def stratified_split(data,seed):
    labels = data['labels']
    train_indices, test_indices, _, _ = train_test_split(
        range(len(labels)), 
        labels, 
        test_size=0.5, 
        stratify=labels,
        random_state=seed  
    )
    # print(train_indices)
    # print(test_indices)

    val_data = { 'id': [], 'features': {name: [] for name in data['features']}, 'labels': [] }
    test_data = { 'id': [], 'features': {name: [] for name in data['features']}, 'labels': [] }

    for idx in train_indices:
        val_data['id'].append(data['id'][idx])
        for name in data['features']:
            val_data['features'][name].append(data['features'][name][idx])
        val_data['labels'].append(data['labels'][idx])

    for idx in test_indices:
        test_data['id'].append(data['id'][idx])
        for name in data['features']:
            test_data['features'][name].append(data['features'][name][idx])
        test_data['labels'].append(data['labels'][idx])

    return val_data, test_data




def load_data(data_path, batch_size, max_len, tokenizer,seed):
    ## split by id
    with open('data/train_id.txt','r') as  f:
        train_lines = f.readlines()
    with open('data/test_id.txt','r') as  f:
        test_lines = f.readlines()
    
    train_id = eval(train_lines[0])
    test_id = eval(test_lines[0])

    dataset = MyDataset(data_path, max_len, tokenizer)

    data_ids = dataset.data['id']

    train_indices = [i for i, data_id in enumerate(data_ids) if data_id in train_id]
    test_indices = [i for i, data_id in enumerate(data_ids) if data_id in test_id]

    train_data = {
        'id': [data_ids[i] for i in train_indices],
        'features': {name: [dataset.data['features'][name][i] for i in train_indices] for name in dataset.data['features']},
        'labels': [dataset.data['labels'][i] for i in train_indices]
    }

    test_data = {
        'id': [data_ids[i] for i in test_indices],
        'features': {name: [dataset.data['features'][name][i] for i in test_indices] for name in dataset.data['features']},
        'labels': [dataset.data['labels'][i] for i in test_indices]
    }

    dataset.data = train_data 
    train_dataset = dataset

    val_data, test_data = stratified_split(test_data,seed)


    val_dataset = MyDataset(data_path=None, max_len=dataset.max_len, tokenizer=dataset.tokenizer)
    val_dataset.data = val_data

    test_dataset = MyDataset(data_path=None, max_len=dataset.max_len, tokenizer=dataset.tokenizer)
    test_dataset.data = test_data



    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(len(train_dataset))
    print(len(test_dataset))
    print(len(val_dataset))

    return train_loader, test_loader, val_loader 


class MyDataset(Dataset):
    def __init__(self, data_path, max_len, tokenizer):
        if data_path is not None:
            with open(data_path, 'rb') as file:
                self.data = pickle.load(file)
        
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.exp_len = 20

    def text_process(self, text, max_len):
        output = self.tokenizer(text,
                                 max_length=max_len,
                                 padding='max_length',
                                 truncation=True)  
        input_ids = output['input_ids']
        attention_mask = output['attention_mask']
        return torch.tensor(input_ids), torch.tensor(attention_mask)

    def __getitem__(self, index):
        label = self.data['labels'][index]
        audio_feature = self.data['features']['audio'][index]
        face_feature = self.data['features']['face'][index]
        posture_feature = self.data['features']['posture'][index]
        video_feature = self.data['features']['video'][index]

        speech = self.data['features']['speech'][index]
        input_id, attention_mask = self.text_process(speech,self.max_len)

        exp = self.data['features']['exp'][index].lower()
        exp_id, _ = self.text_process(exp,self.exp_len)

        exp_id = [(l if l != self.tokenizer.pad_token_id else -100) for l in exp_id]
        exp_id = torch.tensor([l for l in exp_id], dtype=torch.long)
        

        return input_id, attention_mask, audio_feature, face_feature, posture_feature, video_feature, exp_id, label

    def __len__(self):
        return len(self.data['labels'])



