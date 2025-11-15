# metrics.py

import torch.nn.functional as F

def compute_avg_loss(batch_losses):
    return sum(batch_losses) / len(batch_losses) if batch_losses else None

def compute_accuracy(y_prediction, y):
    preds = y_prediction.argmax(dim=1)
    correct = (preds == y).sum().item()
    total = y.numel()
    return correct / total if total > 0 else None   # operand "/" makes float

def compute_avg_accuracy(batch_accuracies):
    return sum(batch_accuracies) / len(batch_accuracies) if batch_accuracies else None

def logits_batch_to_probs(logits_batch):
    return F.softmax(logits_batch, dim=1)
