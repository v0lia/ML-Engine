import torch

from src.utils import visualizer, metrics

def evaluate(dataloader, model, loss_fn, config, device, writer):
    images, _ = next(iter(dataloader))
    visualizer.add_sample_grid(images, writer, tag="Eval sample grid")
    visualizer.add_embedding(dataloader, writer, tag="Eval data embedding")

    print("Starting evaluation...\n")
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
    
    print(f"Average loss: {avg_loss:.4f} | Accuracy: {avg_acc:.3f}")    
    print ("\nFinished evaluation.")
    return


'''
logger
'''

'''
with Timer():
    evaluate(model, dataloader)
'''