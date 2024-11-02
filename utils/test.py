import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

from src.edge_opt import TextClassificationDataset

def test_model(model, dataset, args):
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    dataloader = DataLoader(TextClassificationDataset(dataset, tokenizer, args), batch_size=args.test_bs)
    
    total, correct = 0, 0
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            labels = batch['labels'].to(args.device)
                        
            # Forward pass through the model
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            
            predictions = torch.argmax(logits, dim=-1)
            
            # print(f'Logits: {logits}')
            # print(f'labels: {labels}')
            # print(f'predictions: {predictions}')
            
            # Update the number of correct predictions
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # Update total loss
            total_loss += loss.item() * labels.size(0)
    
    # Calculate accuracy and average loss
    accuracy = correct / total * 100
    avg_loss = total_loss / total
    
    return accuracy, avg_loss
