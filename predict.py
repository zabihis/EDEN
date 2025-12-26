# -*- coding: utf-8 -*-
"""
Title: DEN: Multiscale Expected Density of Nucleotide Encoding for Enhanced DNA Sequence Classification with Hybrid Deep Learning
Author: Saman Zabihi*, Sattar Hashemi, Eghbal Mansoori
Date: December 2025
Description: Implementation script for the EDEN encoding model.
License: MIT License
"""

import os
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Internal Module Imports
from models import Hybrid_CNN
from utils import loadFile_gue, EDNencoderMultiScale, evaluate_model, print_metrics

def main():
    # 1. Setup Command Line Arguments
    parser = argparse.ArgumentParser(description="EDEN Model Prediction CLI")
    
    parser.add_argument('--dataset', type=str, required=True, 
                        help='Name of the dataset folder (e.g., human_prom_core_tata)')
    parser.add_argument('--limit', type=int, required=True, 
                        help='Sequence length limit (seqLenLimit)')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Batch size for inference (default: 64)')
    
    args = parser.parse_args()

    # 2. Environment Configuration
    cur_path = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Construct Paths
    pretrained_weights = os.path.join(cur_path, 'models', f'Hybrid_CNN_{args.dataset}.pth')
    csv_path = os.path.join(cur_path, "datasets", args.dataset, "test.csv")

    # 3. Validation
    if not os.path.exists(pretrained_weights):
        print(f"Error: Weights file not found at {pretrained_weights}")
        return
    if not os.path.exists(csv_path):
        print(f"Error: Dataset file not found at {csv_path}")
        return

    # 4. Load and Encode Data
    print(f'[*] Loading data for {args.dataset}...')
    seq_test, Y_test = loadFile_gue(csv_path)
    
    print(f'[*] Encoding sequences (Limit: {args.limit})...')
    X_test = EDNencoderMultiScale(seq_test, args.limit)

    # 5. Prepare DataLoader
    test_dataset = TensorDataset(torch.from_numpy(X_test).to(torch.float32), 
                                 torch.LongTensor(Y_test))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 6. Initialize and Load Model
    print(f'[*] Initializing Hybrid_CNN on {device}...')
    model = Hybrid_CNN(seq_len=args.limit).to(device)
    
    model.load_state_dict(torch.load(pretrained_weights, 
                                     map_location=device, 
                                     weights_only=True))

    # 7. Evaluation
    print('\n[*] Performing Inference...')
    _, test_metrics = evaluate_model(model, test_loader, nn.BCEWithLogitsLoss(), device)
    
    print('\n' + '='*30)
    print(f'Final Performance: {args.dataset}')
    print('='*30)
    print_metrics(test_metrics, prefix='Test ')

if __name__ == "__main__":
    main()