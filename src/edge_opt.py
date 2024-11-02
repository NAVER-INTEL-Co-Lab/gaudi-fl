import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from transformers import AutoTokenizer

import numpy as np
import habana_frameworks.torch.core as htcore

class TextClassificationDataset(Dataset):
    def __init__(self, dataset, tokenizer, args):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = args.max_seq_length
        self.unique_classes = set()

        for _, row in dataset.iterrows(): 
            text = row['medical_abstract']  
            label = int(row['condition_label'])  
            self.unique_classes.add(label)

            encoding = self.tokenizer.encode_plus(
                text,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt',
            )

            input_ids = encoding['input_ids'].squeeze(0) 
            attention_mask = encoding['attention_mask'].squeeze(0)

            self.examples.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': torch.tensor(label, dtype=torch.long),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
    
    def get_unique_classes(self):
        return list(self.unique_classes)

class EdgeOpt(object):
    def __init__(self, args, train_model=None, edge_dataset=None):
        self.args = args
        self.device = args.device
        self.loss_func = nn.CrossEntropyLoss()
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
        self.model = train_model

        self.dataset = TextClassificationDataset(edge_dataset, self.tokenizer, args)
        if len(self.dataset) == 0:
            raise ValueError("The dataset is empty, cannot initialize DataLoader.")
        self.dataloader = DataLoader(self.dataset, batch_size=args.local_bs, shuffle=True, drop_last=True)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)

        self.unique_classes = self.dataset.get_unique_classes()
        
    def restricted_softmax(self, logits):

        m_logits = torch.ones_like(logits).to(self.device) * self.args.fedrs_alpha
        class_mask = torch.tensor([c - 1 for c in self.unique_classes], dtype=torch.long).to(self.device)
        m_logits[:, class_mask] = 1.0
        logits = logits * m_logits
        
        return logits


    def train(self, global_net=None):
        self.model.train()
        local_ep = np.random.randint(self.args.min_le, self.args.max_le + 1)
        
        for _ in range(local_ep):
            for batch in self.dataloader:
                input_ids = batch['input_ids'].to(self.device)                          
                attention_mask = batch['attention_mask'].to(self.device) 
                labels = batch['labels'].to(self.device)                 

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits
                
                if self.args.method == 'fedrs':
                    logits = self.restricted_softmax(logits)
                
                loss = self.loss_func(logits, labels)

                if self.args.method == 'fedprox':
                    prox_term = 0.0
                    for w, w_t in zip(self.model.parameters(), global_net.parameters()):
                        prox_term += ((w - w_t) ** 2).sum()
                    loss += 0.5 * self.args.lamb * prox_term
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                htcore.mark_step()
                
                self.optimizer.step()
                htcore.mark_step()

        return self.model.state_dict()
