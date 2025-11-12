# import torch

from src.utils import visualizer, metrics

def train(dataloader, model, loss_fn, optimizer, config, device, writer):
    images, _ = next(iter(dataloader))  # images, labels
    visualizer.add_sample_grid(images, writer, tag="Train sample grid")
    visualizer.add_model_graph(model, images[:1], device, writer)
    visualizer.add_embedding(dataloader, writer, tag="Train data embedding")

    print("Starting training...\n")

    model.to(device)
    model.train()   # set model to training mode
    epochs = config.get("epochs", 4)

    global_batch_n = 0     # global batch counter

    for epoch in range(epochs):
        print(f"Start epoch {epoch+1} -------")
        epoch_loss, epoch_acc, global_batch_n = train_one_epoch(dataloader, model, loss_fn, optimizer, device, writer, global_batch_n)
        
        visualizer.add_scalar(writer, "Train/loss/epoch", epoch_loss, epoch)
        visualizer.add_scalar(writer, "Train/accuracy/epoch", epoch_acc, epoch)
        #visualizer.add_prediction_grid(model, images.to(device), labels.to(device), writer, step=epoch)   # TBD someday
        
        print(f"Average loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.3f}")
        print(f"End epoch {epoch+1} ---------\n")
    
    print ("\nFinished training.")
    return

def train_one_epoch(dataloader, model, loss_fn, optimizer, device, writer, global_batch_n):
    batch_losses = []
    batch_accuracies = []

    model.train()
    for (X,y) in dataloader:        # for batch
        X = X.to(device)
        y = y.to(device)
        y_prediction = model(X)     # forward pass
                                    # Backpropogation
        loss = loss_fn(y_prediction, y) # compute loss
        loss.backward()                 # compute grads wrt weights and biases: L(W), L(b)
        optimizer.step()                # adjust weights and biases
        optimizer.zero_grad()           # by default grads add up
        
        batch_loss = loss.item()
        batch_acc = metrics.compute_accuracy(y_prediction, y)

        batch_losses.append(batch_loss)
        batch_accuracies.append(batch_acc)

        visualizer.add_scalar(writer, "Train/loss/batch", batch_loss, global_batch_n)
        visualizer.add_scalar(writer, "Train/accuracy/batch", batch_acc, global_batch_n)
        global_batch_n += 1

    epoch_loss = metrics.compute_avg_loss(batch_losses)
    epoch_acc =  metrics.compute_avg_accuracy(batch_accuracies)
    return epoch_loss, epoch_acc, global_batch_n 

'''
from src.utils.logger import get_logger
logger = get_logger("train")
logger.info("Starting training...")
'''

'''
with Timer():
    evaluate(model, dataloader)
'''