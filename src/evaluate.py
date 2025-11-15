# evaluate.py

import torch

from src.utils import visualizer, metrics, timer
from src.utils.logger import get_logger

def evaluate(dataloader, model, loss_fn, config, device, writer):
    logger = get_logger()
    logger.info(f"Starting evaluation...")

    images, _ = next(iter(dataloader))
    visualizer.add_sample_grid(images, writer, tag="Eval sample grid")
    visualizer.add_embedding(dataloader, writer, tag="Eval data embedding")

    with timer.Timer(name="evaluation", logger=logger):
        batch_losses = []
        batch_accuracies = []

        probs = []
        labels = []

        model.to(device)
        model.eval()

        with torch.no_grad():   # no need to compute grads on test phase
            for X, y in dataloader:
                X = X.to(device)
                y = y.to(device)
                y_prediction = model(X)

                loss = loss_fn(y_prediction, y)
                batch_loss = loss.item()
                batch_acc = metrics.compute_accuracy(y_prediction, y)

                batch_losses.append(batch_loss)
                batch_accuracies.append(batch_acc)

                probs_batch = metrics.logits_batch_to_probs(y_prediction)
                probs.append(probs_batch)
                labels.append(y)

    avg_loss = metrics.compute_avg_loss(batch_losses)
    avg_acc =  metrics.compute_avg_accuracy(batch_accuracies)

    visualizer.add_scalar(writer, "Eval/loss", avg_loss, step=None)
    visualizer.add_scalar(writer, "Eval/accuracy", avg_acc, step=None)
    #visualizer.add_prediction_grid(model, images.to(device), labels.to(device), writer, step=None)   # TBD someday
    
    test_probs = torch.cat(probs)
    test_labels = torch.cat(labels)

    visualizer.add_pr_curves(test_probs, test_labels, writer, step=None)
    
    logger.info(f"Finished evaluation! |Accuracy: {avg_acc:.3f} | Average loss: {avg_loss:.4f}")
    return
