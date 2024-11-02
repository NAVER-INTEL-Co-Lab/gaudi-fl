import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    
    # federated learning arguments
    parser.add_argument('--method', type=str, default='fedavg', help="aggregation method")
    parser.add_argument('--global_ep', type=int, default=100, help="total number of communication rounds")
    parser.add_argument('--num_edges', type=int, default=100, help="number of edge devices")
    parser.add_argument('--frac', type=float, default=0.1, help="fraction of edge devices")
    parser.add_argument('--local_ep', type=int, default=5, help="number of local epochs")
    parser.add_argument('--min_le', type=int, default=1, help="minimum number of local epoch")
    parser.add_argument('--max_le', type=int, default=5, help="maximum number of minimum local epoch")
    parser.add_argument('--local_bs', type=int, default=64, help="local batch size")
    parser.add_argument('--test_bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=1e-5, help="edge device learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.001, help="edge device learning rate hyper parameter")

    # fedprox, fedrs, feddyn arguments
    parser.add_argument('--fedrs_alpha', type=float, default=0.5, help='hyper parameter for fedrs')
    parser.add_argument('--mu', type=float, default=1e-1, help='hyper parameter for feddyn')
    parser.add_argument('--lamb', type=float, default=1e-5, help='hyper parameter for fedprox')
    parser.add_argument('--moon_mu', type=float, default=1e-1, help='hyper parameter for moon')
    parser.add_argument('--beta', type=float, default=0.9, help='hyper parameter for fedcm')

    # plugin agruments
    parser.add_argument('--plugin', action='store_true', help='apply plugin')
    parser.add_argument('--syn', type=str, default='llama_3_2_3b.csv', help='pre-trained model name or path')
    
    # other arguments
    parser.add_argument('--dirichlet_alpha', type=float, default=1e-1, help='hyper parameter for noniid')
    parser.add_argument('--dataset', type=str, default='medical tc', help="name of dataset")
    parser.add_argument('--num_data', type=int, default=1500, help="number of data per subject name")
    parser.add_argument('--num_val', type=int, default=200, help="number of data per class for validation data")
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--tsboard', action='store_true', help='tensorboard')
    parser.add_argument('--debug', action='store_true', help='debug')
    parser.add_argument('--earlystop', action='store_true', help='early stopping option')
    parser.add_argument('--patience', type=int, default=6, help="hyperparameter of early stopping")
    parser.add_argument('--delta', type=float, default=0.01, help="hyperparameter of early stopping")
    parser.add_argument('--model_name_or_path', type=str, default='dmis-lab/biobert-v1.1', help='pre-trained model name or path')
    parser.add_argument('--max_seq_length', type=int, default=512, help='Maximum sequence length')
    
    args = parser.parse_args()
    
    return args
