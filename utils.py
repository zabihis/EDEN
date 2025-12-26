import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import KernelDensity
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, confusion_matrix

def findstr(string, ch):
    for i, ltr in enumerate(string):
        if ltr == ch:
            yield i


def EDNcalc(idxPoints, bandwidth, kernel, xspace):
    if len(idxPoints) > 0:
        kde_model = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(idxPoints)
        EDN_sig = np.exp(kde_model.score_samples(xspace)) * len(idxPoints)
    else:
        EDN_sig = np.zeros((xspace.shape[0]))
    return EDN_sig


def EDNencoderMultiScale(seqs, seqLenLimit):
    kernel = "cosine"
    bandwidthList = [0.5, 1.5, 3, 4.5]
    Nseq, Nscale = len(seqs), len(bandwidthList)
    seq_len = len(seqs[0])
    xspace = np.linspace(0, seqLenLimit-1, seqLenLimit)[:, np.newaxis]
    seqsEncoded = np.zeros((Nseq, 4, seqLenLimit, Nscale), dtype='float32')

    if abs(seqLenLimit-seq_len) > 1:
        print(f'* ATTENTION: Sequence length is {seq_len}, limit is {seqLenLimit}.')

    for iseq in range(Nseq):
        one_seq = seqs[iseq].upper().replace("U", "T")
        if len(one_seq) > seqLenLimit:
            one_seq = one_seq[:seqLenLimit]

        for iscale, bandwidth in enumerate(bandwidthList):
            sigs = []
            for base in ['A', 'T', 'G', 'C']:
                idx = np.array(list(findstr(one_seq, base)))[:, np.newaxis]
                sigs.append(EDNcalc(idx, bandwidth, kernel, xspace))
            seqsEncoded[iseq, :, :, iscale] = np.array(sigs)
            
    print(f'EDN encoding done! {Nseq} x {seqLenLimit}')
    return seqsEncoded


def loadFile_gue(file_path):
    df = pd.read_csv(file_path)
    print(f"File loaded! Data size: {df.shape}")
    return df['sequence'].to_numpy(), df['label'].to_numpy()

def evaluate_model(model, data_loader, criterion, device):
    """Evaluate model and return loss and metrics"""
    model.eval()
    running_loss = 0.0
    all_targets, all_probs, all_preds = [], [], []

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            running_loss += criterion(outputs, target.float().unsqueeze(1)).item()

            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs)
            all_preds.extend((probs > 0.5).astype(int))

    loss = running_loss / len(data_loader)
    metrics = calculate_metrics(np.array(all_targets), np.array(all_preds), np.array(all_probs))
    return loss, metrics


def calculate_metrics(y_true, y_pred, y_prob):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_prob),
        'mcc': matthews_corrcoef(y_true, y_pred),
    }


def print_metrics(metrics, prefix=''):
    print(f"{prefix}Accuracy: {metrics['accuracy']:.4f} | "
          f"AUC: {metrics['auc']:.4f} | MCC: {metrics['mcc']:.4f}")