import os
import pytorch_lightning as pl
import torch.optim as optim

from utils import torch, cuda, device, plot_dataset_sample, test, train, plot_model_performance, test_acc, \
    plot_grad_cam, load_model_from_checkpoint, plot_missclassified_preds, save_model
from dataset import get_loader, dataset_mean, dataset_std, CustomCIFARR10LightningDataModule
from models.resnet import ResNet18, nn
from torchsummary import summary
from torch_lr_finder import LRFinder


model = ResNet18()
batch_size = 512
kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}
train_loader, test_loader = get_loader(**kwargs)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-2)


def init(network=None, show_sample=True, show_model_summary=True, find_lr=False, start_train=False, resume=False):

    global model

    if network:
        model = network

    if isinstance(model, pl.LightningModule):

        batch_size = 64
        kwargs = {'batch_size': batch_size, 'shuffle': True, 's10': True, 'num_workers': os.cpu_count(), 'pin_memory': True}
        train_dataloader, test_dataloader = get_loader(**kwargs)

        if show_sample:
            plot_dataset_sample(train_dataloader, dataset_mean, dataset_std)

        model.is_find_max_lr = find_lr

        if start_train:
            trainer = pl.Trainer(
                precision=16,
                max_epochs=24
            )
            trainer.fit(model, train_dataloader, test_dataloader)

            save_model(model)

        if resume:
            chk = load_model_from_checkpoint()
            model.load_state_dict(chk['model'])




    elif isinstance(model, nn.Module):

        model.to(device)
        if show_sample:
            plot_dataset_sample(train_loader, dataset_mean, dataset_std)

        if show_model_summary:
            summary(model, input_size=(3, 32, 32))

        if find_lr:
            lr_finder = LRFinder(model, optimizer, criterion, device=device)
            lr_finder.range_test(train_loader, end_lr=100, num_iter=100)
            lr_finder.plot()  # to inspect the loss-learning rate graph
            lr_finder.reset()  # to reset the model and optimizer to their initial state


        if start_train:
            train_model(resume=resume)


def train_model(start_epoch=1, resume=False, num_epochs=20):

    steps_per_epoch = len(train_loader)
    # scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=3.20E-06, max_lr=3.20E-04, step_size_up=5*steps_per_epoch, step_size_down=19*steps_per_epoch, cycle_momentum=False, verbose=False)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                                              max_lr=1.83E-04,
                                              epochs=num_epochs,
                                              steps_per_epoch=steps_per_epoch,
                                              pct_start=0.3,
                                              div_factor=100,
                                              final_div_factor=100,
                                              three_phase=False,
                                              verbose=False
                                              )
    if resume:
        checkpoint = load_model_from_checkpoint()
        model.load_state_dict(checkpoint['model']).to(device)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']


    for epoch in range(start_epoch, num_epochs + 1):
        print(f'Epoch {epoch} LR {scheduler.get_last_lr()}')
        train(model, device, train_loader, optimizer, criterion, scheduler)
        test(model, device, test_loader, criterion)



    #plot_grad_cam(model, dataset_mean, dataset_std, count=10, missclassified=True)
    #plot_model_performance()



#!git clone https://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@github.com/piygr/s11erav1.git