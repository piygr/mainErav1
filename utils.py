import torch
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
import numpy as np
from dataset import get_data_label_name
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import os

# CUDA?
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)
device = torch.device("cuda" if cuda else "cpu")

def denormalize(img, mean, std):
    MEAN = torch.tensor(mean)
    STD = torch.tensor(std)

    img = img * STD[:, None, None] + MEAN[:, None, None]
    i_min = img.min().item()
    i_max = img.max().item()

    img_bar = (img - i_min)/(i_max - i_min)

    return img_bar

def plot_dataset_sample(data_loader, mean, std):
    batch_data, batch_label = next(iter(data_loader))


    # fig = plt.figure()
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        #plt.tight_layout()
        x = denormalize(batch_data[i].cpu(), mean, std)

        image = np.array(255 * x, np.int16).transpose(1, 2, 0)
        plt.imshow(image, vmin=0, vmax=255)
        plt.xticks([])
        plt.yticks([])
        plt.title(get_data_label_name(batch_label[i].item()), fontsize=8)


def plot_missclassified_preds(mean, std, count=20):

    for i in range(count):
        plt.subplot(int(count / 5), 5, i + 1)
        # plt.tight_layout()
        x = denormalize(test_incorrect_pred['images'][i].cpu(), mean, std)

        image = np.array(255 * x, np.int16).transpose(1, 2, 0)
        plt.imshow(image, vmin=0, vmax=255, cmap='gray')

        plt.xticks([])
        plt.yticks([])

        title = '✅' + get_data_label_name(test_incorrect_pred['ground_truths'][i].item()) + ' / ❌' + \
                get_data_label_name(test_incorrect_pred['predicted_vals'][i].item())
        plt.title(title, fontsize=8)


# code to move any list dict of tensor to the cuda/cpu
def move_to(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    else:
        raise TypeError("Invalid type for move_to")



def plot_grad_cam(model, mean, std, count=20, missclassified=True, target_layers=None):
    if not target_layers:
        target_layers = [model.layer4[-1]]

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=0)

    fig = plt.figure()
    for i in range(count):
        plt.subplot(int(count / 5), 5, i + 1)
        plt.tight_layout()
        if not missclassified:
            pred_dict = test_correct_pred
        else:
            pred_dict = test_incorrect_pred

        targets = [ClassifierOutputTarget(pred_dict['ground_truths'][i].item())]

        grayscale_cam = cam(input_tensor=pred_dict['images'][i][None, :], targets=targets)

        # grayscale_cam = grayscale_cam[0, :].transpose(1, 2, 0)

        x = denormalize(pred_dict['images'][i].cpu(), mean, std)

        image = np.array(255 * x, np.int16).transpose(1, 2, 0)
        img_tensor = np.array(x, np.float16).transpose(1, 2, 0)

        visualization = show_cam_on_image(img_tensor, grayscale_cam.transpose(1, 2, 0), use_rgb=True, image_weight=0.8)

        plt.imshow(image, vmin=0, vmax=255)
        plt.imshow(visualization, alpha=0.6, vmin=0, vmax=255)
        plt.xticks([])
        plt.yticks([])

        title = get_data_label_name(pred_dict['ground_truths'][i].item()) + ' / ' + \
                get_data_label_name(pred_dict['predicted_vals'][i].item())
        plt.title(title, fontsize=8)


# Data to plot accuracy and loss graphs
train_losses = []
test_losses = []
train_acc = []
test_acc = []

test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}
test_correct_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}

def get_correct_pred_count(pPrediction, pLabels):
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()


def add_predictions(data, pred, target):
    diff_preds = pred.argmax(dim=1) - target
    for idx, d in enumerate(diff_preds):
        if d.item() != 0:
            test_incorrect_pred['images'].append(data[idx])
            test_incorrect_pred['ground_truths'].append(target[idx])
            test_incorrect_pred['predicted_vals'].append(pred.argmax(dim=1)[idx])
        elif d.item() == 0:
            test_correct_pred['images'].append(data[idx])
            test_correct_pred['ground_truths'].append(target[idx])
            test_correct_pred['predicted_vals'].append(pred.argmax(dim=1)[idx])


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
    test_correct_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            #print(data.size(), target.size())
            output = model(data)
            local_loss = len(data) * criterion(output, target).item()  # sum up batch loss => mean_loss * batch_size
            test_loss += local_loss

            correct += get_correct_pred_count(output, target)

            add_predictions(data, output, target)

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


def create_full_model_checkpoint(model, optimizer, scheduler, epoch):
    print('Saving..')
    state = {
        'model': model.state_dict(),
        'epoch': epoch,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'test_acc': test_acc,
        'test_losses': test_losses,
        'train_losses': train_losses[-1:],
        'train_acc': train_acc[-1:],
        'test_incorrect_pred': dict(images=test_incorrect_pred['images'][-20:],
                                    ground_truths=test_incorrect_pred['ground_truths'][-20:],
                                    predicted_vals=test_incorrect_pred['predicted_vals'][-20:]),
        'test_correct_pred': dict(images=test_correct_pred['images'][-20:],
                                    ground_truths=test_correct_pred['ground_truths'][-20:],
                                    predicted_vals=test_correct_pred['predicted_vals'][-20:])
    }

    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')

    torch.save(state, './checkpoint/ckpt.pth')


def save_model(model):
    print('Saving..')
    state = dict(model= model.state_dict())
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')

    torch.save(state, './checkpoint/ckpt_light.pth')


def load_model_from_checkpoint():
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    train_losses = checkpoint.get('train_losses', [])
    test_losses = checkpoint.get('test_losses', [])
    train_acc = checkpoint.get('train_acc',[])
    test_acc = checkpoint.get('test_acc',[])
    test_incorrect_pred = checkpoint.get('test_incorrect_pred', dict(images=[], ground_truths=[], predicted_vals=[]))
    test_correct_pred = checkpoint.get('test_correct_pred',  dict(images=[], ground_truths=[], predicted_vals=[]))

    return checkpoint