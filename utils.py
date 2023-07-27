import torch
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
import numpy as np
from dataset import get_data_label_name

# CUDA?
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)

def plot_dataset_sample(data_loader, mean, std):
    batch_data, batch_label = next(iter(data_loader))
    MEAN = torch.tensor(mean)
    STD = torch.tensor(std)

    # fig = plt.figure()
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.tight_layout()
        x = batch_data[i] * STD[:, None, None] + MEAN[:, None, None]

        image = np.array(255 * x, np.int16).transpose(1, 2, 0)
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
        plt.title(get_data_label_name(batch_label[i].item()))


def plot_incorrect_preds(mean, std, count=20):
    MEAN = torch.tensor(mean)
    STD = torch.tensor(std)

    for i in range(count):
        plt.subplot(int(count/5), 5, i + 1)
        plt.tight_layout()
        x = test_incorrect_pred['images'][i] * STD[:, None, None] + MEAN[:, None, None]

        image = np.array(255 * x, np.int16).transpose(1, 2, 0)
        plt.imshow(image)

        plt.xticks([])
        plt.yticks([])

        title = get_data_label_name( test_incorrect_pred['ground_truths'][i].item() ) + ' / ' + \
                get_data_label_name( test_incorrect_pred['predicted_vals'][i].item() )
        plt.title(title)

# Data to plot accuracy and loss graphs
train_losses = []
test_losses = []
train_acc = []
test_acc = []

test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}

def get_correct_pred_count(pPrediction, pLabels):
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()


def add_incorrect_predictions(data, pred, target):
    diff_preds = pred.argmax(dim=1) - target
    for idx, d in enumerate(diff_preds):
        if d.item() != 0:
            test_incorrect_pred['images'].append(data[idx])
            test_incorrect_pred['ground_truths'].append(target[idx])
            test_incorrect_pred['predicted_vals'].append(pred.argmax(dim=1)[idx])


def train(model, device, train_loader, optimizer, criterion, scheduler):
    model.train()
    pbar = tqdm(train_loader)

    train_loss = 0
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Predict
        pred = model(data)

        # Calculate loss
        loss = criterion(pred, target)
        train_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        scheduler.step()

        correct += get_correct_pred_count(pred, target)
        processed += len(data)

        pbar.set_description(
            desc=f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

    train_acc.append(100 * correct / processed)
    train_losses.append(train_loss / len(train_loader))


def test(model, device, test_loader, criterion):
    model.eval()

    test_loss = 0
    correct = 0

    test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            #print(data.size(), target.size())
            output = model(data)
            local_loss = len(data) * criterion(output, target).item()  # sum up batch loss => mean_loss * batch_size
            test_loss += local_loss

            correct += get_correct_pred_count(output, target)

            add_incorrect_predictions(data, output, target)

    test_loss /= len(test_loader.dataset)  # mean of the test_loss post all the batches
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def plot_model_performance():
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
