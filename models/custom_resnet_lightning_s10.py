import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import matplotlib.pyplot as plt
from torch_lr_finder import LRFinder
import numpy as np
from utils import get_correct_pred_count, add_predictions, test_incorrect_pred, test_correct_pred, denormalize

NO_GROUPS = 4
class ResnetBlock(nn.Module):
    def __init__(self, input_channel, output_channel, padding=1, norm='bn', drop=0.01):

        super(ResnetBlock, self).__init__()

        self.conv1 = nn.Conv2d(input_channel, output_channel, 3, padding=padding)

        if norm == 'bn':
            self.n1 = nn.BatchNorm2d(output_channel)
        elif norm == 'gn':
            self.n1 = nn.GroupNorm(NO_GROUPS, output_channel)
        elif norm == 'ln':
            self.n1 = nn.GroupNorm(1, output_channel)

        self.drop1 = nn.Dropout2d(drop)

        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=padding)

        if norm == 'bn':
            self.n2 = nn.BatchNorm2d(output_channel)
        elif norm == 'gn':
            self.n2 = nn.GroupNorm(NO_GROUPS, output_channel)
        elif norm == 'ln':
            self.n2 = nn.GroupNorm(1, output_channel)

        self.drop2 = nn.Dropout2d(drop)


    '''
    Depending on the model requirement, Convolution block with number of layers is applied to the input image
    '''
    def __call__(self, x):

        x = self.conv1(x)
        x = self.n1(x)
        x = F.relu(x)

        x = self.drop1(x)


        #if layers >= 2:

        x = self.conv2(x)

        x = self.n2(x)
        x = F.relu(x)
        x = self.drop2(x)

        return x


class S10LightningModel(pl.LightningModule):
    def __init__(self, base_channels, drop=0.01, loss_function=F.cross_entropy, is_find_max_lr=False, max_lr=3.20E-04):
        super(S10LightningModel, self).__init__()

        self.is_find_max_lr = is_find_max_lr
        self.max_lr = max_lr
        self.criterion = loss_function

        self.metric = dict(train=0,
                        val=0,
                        train_total=0,
                        val_total=0,
                        epoch_train_loss=[],
                        epoch_val_loss=[],
                        train_loss=[],
                        val_loss=[],
                        train_acc=[],
                        val_acc=[])

        self.base_channels = base_channels

        self.prep_layer = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.Dropout2d(drop)
        )

        # layer1
        self.x1 = nn.Sequential(
            nn.Conv2d(base_channels, 2 * base_channels, 3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(2 * base_channels),
            nn.ReLU(),
            nn.Dropout2d(drop)
        )

        self.R1 = ResnetBlock(2 * base_channels, 2 * base_channels, padding=1, drop=drop)

        # layer2
        self.layer2 = nn.Sequential(
            nn.Conv2d(2 * base_channels, 4 * base_channels, 3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(4 * base_channels),
            nn.ReLU(),
            nn.Dropout2d(drop)
        )

        # layer3
        self.x2 = nn.Sequential(
            nn.Conv2d(4 * base_channels, 8 * base_channels, 3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(8 * base_channels),
            nn.ReLU(),
            nn.Dropout2d(drop)
        )

        self.R2 = ResnetBlock(8 * base_channels, 8 * base_channels, padding=1, drop=drop)

        self.pool = nn.MaxPool2d(4)

        self.fc = nn.Linear(8 * base_channels, 10)

    def forward(self, x):

        # print(x.size())

        x = self.prep_layer(x)
        # print(x.size())

        x = self.x1(x)
        # print('x1', x.size())

        x = self.R1(x) + x
        # print('x', x.size())

        x = self.layer2(x)
        # print(x.size())

        x = self.x2(x)
        # print('x2', x.size())

        x = self.R2(x) + x
        # print('x', x.size())

        x = self.pool(x)
        # print(x.size())

        x = x.view(x.size(0), 8 * self.base_channels)
        # print(x.size())

        x = self.fc(x)
        # print(x.size())

        return F.log_softmax(x, dim=1)


    def get_layer(self, idx):
        layers = [self.prep_layer, self.x1, self.layer2, self.x2, self.pool]

        if idx < len(layers) and idx >= 0:
            return layers[idx]


    def training_step(self, train_batch, batch_idx):
        x, target = train_batch
        output = self.forward(x)
        loss = self.criterion(output, target)

        self.metric['train'] += get_correct_pred_count(output, target)
        self.metric['train_total'] += len(x)
        self.metric['epoch_train_loss'].append(loss)

        acc = 100 * self.metric['train'] / self.metric['train_total']

        self.log_dict({'train_loss': loss, 'train_acc': acc})
        return loss


    def validation_step(self, val_batch, batch_idx):
        x, target = val_batch
        output = self.forward(x)
        loss = self.criterion(output, target)

        self.metric['val'] += get_correct_pred_count(output, target)
        self.metric['val_total'] += len(x)
        self.metric['epoch_val_loss'].append(loss)

        acc = 100 * self.metric['val'] / self.metric['val_total']

        if self.current_epoch == self.trainer.max_epochs - 1:
            add_predictions(x, output, target)

        self.log_dict({'val_loss': loss, 'val_acc': acc})


    def test_step(self, test_batch, batch_idx):
        self.validation_step(test_batch, batch_idx)

    def train_dataloader(self):
        if not self.trainer.train_dataloader:
            self.trainer.fit_loop.setup_data()

        return self.trainer.train_dataloader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-6, weight_decay=0.01)
        self.find_lr(optimizer)
        print(self.max_lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                  max_lr=self.max_lr,
                                                  epochs=self.trainer.max_epochs,
                                                  steps_per_epoch=len(self.train_dataloader()),
                                                  pct_start=5 / self.trainer.max_epochs,
                                                  div_factor=100,
                                                  final_div_factor=100,
                                                  three_phase=False,
                                                  verbose=False
                                                  )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                'interval': 'step', # or 'epoch'
                'frequency': 1
            },
        }


    def on_validation_epoch_end(self):
        if self.metric['train_total']:
            print('Epoch ', self.current_epoch)
            train_acc = 100 * self.metric['train'] / self.metric['train_total']
            epoch_loss = sum(self.metric['epoch_train_loss']) / len(self.metric['epoch_train_loss'])
            self.metric['train_loss'].append( epoch_loss.item() )
            self.metric['train_acc'].append(train_acc)


            print('Train Loss: ', epoch_loss.item(), ' Accuracy: ', str(train_acc) + '%', ' [',
                  self.metric['train'], '/', self.metric['train_total'], ']')

            self.metric['train'] = 0
            self.metric['train_total'] = 0
            self.metric['epoch_train_loss'] = []

            val_acc = 100 * self.metric['val'] / self.metric['val_total']

            epoch_loss = sum(self.metric['epoch_val_loss']) / len(self.metric['epoch_val_loss'])
            self.metric['val_loss'].append( epoch_loss.item() )
            self.metric['val_acc'].append(val_acc)

            print('Validation Loss: ', epoch_loss.item(), ' Accuracy: ', str(val_acc) + '%', ' [', self.metric['val'],
                  '/', self.metric['val_total'], ']\n')

            self.metric['val'] = 0
            self.metric['val_total'] = 0
            self.metric['epoch_val_loss'] = []



    def find_lr(self, optimizer):
        if not self.is_find_max_lr:
            return

        lr_finder = LRFinder(self, optimizer, self.criterion)
        lr_finder.range_test(self.train_dataloader(), end_lr=100, num_iter=100)
        _, best_lr = lr_finder.plot()  # to inspect the loss-learning rate graph
        lr_finder.reset()
        self.max_lr = best_lr


    def plot_model_performance(self):
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        axs[0, 0].plot( self.metric['train_loss'] )
        axs[0, 0].set_title("Training Loss")
        axs[1, 0].plot( self.metric['train_acc'] )
        axs[1, 0].set_title("Training Accuracy")
        axs[0, 1].plot( self.metric['val_loss'] )
        axs[0, 1].set_title("Test Loss")
        axs[1, 1].plot( self.metric['val_acc'] )
        axs[1, 1].set_title("Test Accuracy")


    def plot_grad_cam(self, mean, std, target_layers, get_data_label_name, count=10, missclassified=True, grad_opacity=1.0):
        cam = GradCAM(model=self, target_layers=target_layers)

        #fig = plt.figure()
        for i in range(count):
            plt.subplot(int(count / 5), 5, i + 1)
            plt.tight_layout()
            if not missclassified:
                pred_dict = test_correct_pred
            else:
                pred_dict = test_incorrect_pred

            targets = [ClassifierOutputTarget(pred_dict['ground_truths'][i].cpu().item())]

            grayscale_cam = cam(input_tensor=pred_dict['images'][i][None, :].cpu(), targets=targets)

            x = denormalize(pred_dict['images'][i].cpu(), mean, std)

            image = np.array(255 * x, np.int16).transpose(1, 2, 0)
            img_tensor = np.array(x, np.float16).transpose(1, 2, 0)

            visualization = show_cam_on_image(img_tensor, grayscale_cam.transpose(1, 2, 0), use_rgb=True,
                                              image_weight=(1.0 - grad_opacity) )

            plt.imshow(image, vmin=0, vmax=255)
            plt.imshow(visualization, vmin=0, vmax=255, alpha=grad_opacity)
            plt.xticks([])
            plt.yticks([])

            title = get_data_label_name(pred_dict['ground_truths'][i].item()) + ' / ' + \
                    get_data_label_name(pred_dict['predicted_vals'][i].item())
            plt.title(title, fontsize=8)
