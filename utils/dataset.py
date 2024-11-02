import pandas as pd
import numpy as np

def load_data():
    dataset_train = pd.read_csv('./dataset/medical_tc_train.csv')
    dataset_test = pd.read_csv('./dataset/medical_tc_test.csv')
    
    return dataset_train, dataset_test

def get_labels(dataset):
    # Extract 'condition_label' column and convert to integers
    condition_labels = dataset['condition_label'].astype(int).tolist()  # Access column directly
    condition_label_list = sorted(set(condition_labels))
    condition_label_to_label = {label: idx for idx, label in enumerate(condition_label_list)}
    labels = np.array([condition_label_to_label[label] for label in condition_labels])
    return labels, condition_label_to_label

def noniid(dataset, args):
    labels, _ = get_labels(dataset)
    unique_labels = np.unique(labels)
    dict_users = {i: [] for i in range(args.num_edges)}
    idxs = np.arange(len(dataset))

    # Partition data per class using a Dirichlet distribution
    for c in unique_labels:
        idx_c = idxs[labels == c]
        np.random.shuffle(idx_c)
        
        # Ensure proportions do not contain zeros by applying a minimum threshold
        proportions = np.random.dirichlet(np.repeat(args.dirichlet_alpha, args.num_edges))
        
        # Apply a small threshold to avoid 0 values in proportions
        epsilon = 1e-3  # Small value to replace near-zero proportions
        proportions = np.maximum(proportions, epsilon)
        proportions = proportions / proportions.sum()  # Re-normalize to ensure sum is 1
        
        counts = (proportions * len(idx_c)).astype(int)
        # Adjust counts to ensure all samples are allocated
        counts[-1] = len(idx_c) - counts[:-1].sum()
        start = 0
        for i in range(args.num_edges):
            count = counts[i]
            dict_users[i].extend(idx_c[start:start + count])
            start += count
    
    return dict_users

def create_edge_datasets(dataset, dict_users):
    edge_datasets = {}
    for edge_id, indices in dict_users.items():
        edge_dataset = dataset.iloc[indices]  # Use iloc to index rows in a DataFrame
        edge_datasets[edge_id] = edge_dataset
    return edge_datasets

def partition_data(dataset_train, args):
    # Extract 'condition_label' labels and assign integer labels    
    dict_users = noniid(dataset_train, args)
    edge_datasets = create_edge_datasets(dataset_train, dict_users)
    
    return edge_datasets