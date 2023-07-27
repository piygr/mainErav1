import torch.optim as optim
from utils import torch, cuda, plot_dataset_sample, test, train, plot_model_performance, test_acc
from dataset import get_loader, dataset_mean, dataset_std
from models.resnet import ResNet18, nn
from torchsummary import summary
from torch_lr_finder import LRFinder

device = torch.device("cuda" if cuda else "cpu")
model = ResNet18().to(device)

batch_size = 512
kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}
train_loader, test_loader = get_loader(**kwargs)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-2)


def init(show_sample=True, show_model_summary=True, find_lr=False):

    if show_sample:
        plot_dataset_sample(train_loader, dataset_mean, dataset_std)

    if show_model_summary:
        summary(model, input_size=(3, 32, 32))

    if find_lr:
        lr_finder = LRFinder(model, optimizer, criterion, device=device)
        lr_finder.range_test(train_loader, end_lr=100, num_iter=100)
        lr_finder.plot()  # to inspect the loss-learning rate graph
        lr_finder.reset()  # to reset the model and optimizer to their initial state


def train_model():
    num_epochs = 20
    steps_per_epoch = len(train_loader)
    # scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=3.20E-06, max_lr=3.20E-04, step_size_up=5*steps_per_epoch, step_size_down=19*steps_per_epoch, cycle_momentum=False, verbose=False)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                                              max_lr=1.83E-04,
                                              epochs=num_epochs,
                                              steps_per_epoch=steps_per_epoch,
                                              pct_start=5 / num_epochs,
                                              div_factor=100,
                                              final_div_factor=100,
                                              three_phase=False,
                                              verbose=False
                                              )

    for epoch in range(1, num_epochs + 1):
        print(f'Epoch {epoch} LR {scheduler.get_last_lr()}')
        train(model, device, train_loader, optimizer, criterion, scheduler)
        test(model, device, test_loader, criterion)

    plot_model_performance()



#!git clone https://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@github.com/piygr/s11erav1.git