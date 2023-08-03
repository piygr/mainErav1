import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_lr_finder import LRFinder
from utils import train_losses, test_losses, train_acc, test_acc, test_correct_pred, test_incorrect_pred, \
    get_correct_pred_count, add_predictions


class ResnetBlock(pl.LightningModule):
    def __init__(self, input_channel, output_channel, padding=1, drop=0.01):

        super(ResnetBlock, self).__init__()

        self.conv1 = nn.Conv2d(input_channel, output_channel, 3, padding=padding)

        self.n1 = nn.BatchNorm2d(output_channel)


        self.drop1 = nn.Dropout2d(drop)

        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=padding)

        self.n2 = nn.BatchNorm2d(output_channel)

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

        self.acc = dict(train=0, val=0, train_total=0, val_total=0)


        self.base_channels = base_channels

        self.prep_layer = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.Dropout2d(drop)
        )

        #layer1
        self.x1 = nn.Sequential(
            nn.Conv2d(base_channels, 2*base_channels, 3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(2*base_channels),
            nn.ReLU(),
            nn.Dropout2d(drop)
        )

        self.R1 = ResnetBlock(2*base_channels, 2*base_channels, padding=1, drop=drop)

        #layer2
        self.layer2 = nn.Sequential(
            nn.Conv2d(2*base_channels, 4*base_channels, 3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(4*base_channels),
            nn.ReLU(),
            nn.Dropout2d(drop)
        )

        #layer3
        self.x2 = nn.Sequential(
            nn.Conv2d(4*base_channels, 8*base_channels, 3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(8*base_channels),
            nn.ReLU(),
            nn.Dropout2d(drop)
        )

        self.R2 = ResnetBlock(8*base_channels, 8*base_channels, padding=1, drop=drop)

        self.pool = nn.MaxPool2d(4)

        self.fc = nn.Linear(8*base_channels, 10)

    def forward(self, x):


        x = self.prep_layer(x)

        x = self.x1(x)

        x = self.R1(x) + x

        x = self.layer2(x)

        x = self.x2(x)

        x = self.R2(x) + x

        x = self.pool(x)

        x = x.view(x.size(0), 8*self.base_channels)

        x = self.fc(x)

        return F.log_softmax(x, dim=1)


    def get_layer(self, idx):
        layers = [self.prep_layer, self.x1, self.R1, self.layer2, self.x2, self.R2, self.pool]

        if idx < len(layers) and idx >= 0:
            return layers[idx]
        

    def training_step(self, train_batch, batch_idx):
        x, target = train_batch
        output = self.forward(x)
        loss = self.criterion(output, target)

        self.acc['train'] += get_correct_pred_count(output, target)
        self.acc['train_total'] += len(x)

        self.log_dict({'train_loss': loss, 'acc': 100*self.acc['train']/self.acc['train_total']})
        return loss


    def validation_step(self, val_batch, batch_idx):
        x, target = val_batch
        output = self.forward(x)
        loss = self.criterion(output, target)

        self.acc['val'] += get_correct_pred_count(output, target)
        self.acc['val_total'] += len(x)

        if self.current_epoch == self.trainer.max_epochs - 1:
            add_predictions(x, output, target)

        self.log_dict({'val_loss': loss, 'acc': 100 * self.acc['val'] / self.acc['val_total']})


    def train_dataloader(self):
        if not self.trainer.train_dataloader:
            self.trainer.fit_loop.setup_data()

        return self.trainer.train_dataloader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-6, weight_decay=0.01)
        self.find_lr(optimizer)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                  max_lr=self.max_lr,
                                                  epochs=self.trainer.max_epochs,
                                                  total_steps=self.trainer.estimated_stepping_batches,
                                                  pct_start=5 / self.trainer.max_epochs,
                                                  div_factor=100,
                                                  final_div_factor=100,
                                                  three_phase=False,
                                                  verbose=False
                                                  )
        return [optimizer], [scheduler]


    '''def on_train_epoch_end(self):
       print('Epoch ', self.current_epoch, ' Train Accuracy', 100*self.acc['train']/self.acc['train_total'], '% [', self.acc['train'], '/', self.acc['train_total'], ']')
       self.acc['train'] = 0
       self.acc['train_total'] = 0'''


    def on_validation_epoch_end(self):
        if self.acc['train_total']:
            print('Epoch ', self.current_epoch)
            print('Train Accuracy', 100 * self.acc['train'] / self.acc['train_total'], '% [',
                  self.acc['train'], '/', self.acc['train_total'], ']')
            self.acc['train'] = 0
            self.acc['train_total'] = 0

            print('Validation Accuracy', 100 * self.acc['val'] / self.acc['val_total'], '% [', self.acc['val'], '/', self.acc['val_total'], ']')
            self.acc['val'] = 0
            self.acc['val_total'] = 0



    def find_lr(self, optimizer):
        if not self.is_find_max_lr:
            return self.max_lr

        lr_finder = LRFinder(self, optimizer, self.criterion)
        lr_finder.range_test(self.train_dataloader(), end_lr=100, num_iter=100)
        _, best_lr = lr_finder.plot()  # to inspect the loss-learning rate graph
        lr_finder.reset()
        self.max_lr = best_lr