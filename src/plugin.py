import pandas as pd
import numpy as np

def calculate_data_shortage(class_counts, class_label):
    if class_label not in class_counts:
        return 0  
    max_class_count = class_counts.max()  
    class_count = class_counts[class_label] 
    return max_class_count, class_count

def generate_synthetic_data(edge_data, paraphrased_data, class_label, num_samples_to_add):

    class_rows = paraphrased_data[paraphrased_data['condition_label'] == class_label]
    
    original_indices = edge_data.index
    matched_rows = class_rows[class_rows.index.isin(original_indices)]
    return matched_rows.sample(n=num_samples_to_add, replace=True) if len(matched_rows) >= num_samples_to_add else matched_rows

def balance_data_per_edge(edge_data, paraphrased_data):

    class_counts = edge_data['condition_label'].value_counts()  
    max_class_count = class_counts.max()  

    for class_label in class_counts.index:
        max_class_count, class_count = calculate_data_shortage(class_counts, class_label)
        
        if class_count * 2 > max_class_count:
            num_samples_to_add = max_class_count - class_count
            if num_samples_to_add > 0:
                synthetic_data = generate_synthetic_data(edge_data, paraphrased_data, class_label, num_samples_to_add)
                if synthetic_data is not None:
                    edge_data = pd.concat([edge_data, synthetic_data]) 

        else:
            num_samples_to_add = class_count
            synthetic_data = generate_synthetic_data(edge_data, paraphrased_data, class_label, num_samples_to_add)
            if synthetic_data is not None:
                edge_data = pd.concat([edge_data, synthetic_data])

    return edge_data

def apply_iid_approximation(edge_datasets, paraphrased_file):
    
    paraphrased_data = pd.read_csv(paraphrased_file)

    for i in range(len(edge_datasets)):
        edge_datasets[i] = balance_data_per_edge(edge_datasets[i], paraphrased_data)

    return edge_datasets

def calculate_weighted_global_distribution(edge_datasets):
    
    global_counts = pd.Series(dtype=float)
    total_samples = 0
    
    for i in range(len(edge_datasets)):
        class_counts = edge_datasets[i]['condition_label'].value_counts()
        dataset_size = len(edge_datasets[i])
        global_counts = global_counts.add(class_counts * dataset_size, fill_value=0)
        total_samples += dataset_size
    
    global_distribution = global_counts / total_samples
    
    return global_distribution

def distance_between_distributions(distribution1, distribution2):

    combined = pd.concat([distribution1, distribution2], axis=1, sort=True).fillna(0)
    return np.linalg.norm(combined.iloc[:, 0] - combined.iloc[:, 1])  # 유클리드 거리 계산

def select_edge_devices(edge_datasets, global_distribution, top_n):

    distances = []
    
    for i in range(len(edge_datasets)):
        local_distribution = edge_datasets[i]['condition_label'].value_counts(normalize=True)
        distance = distance_between_distributions(local_distribution, global_distribution)
        distances.append((i, distance))
    
    selected_devices = sorted(distances, key=lambda x: x[1])[:top_n]
    return [idx for idx, _ in selected_devices]

def balanced_sampling(edge_datasets, top_n):
    
    global_distribution = calculate_weighted_global_distribution(edge_datasets)
    selected_devices = select_edge_devices(edge_datasets, global_distribution, top_n)
    
    return selected_devices