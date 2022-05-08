import sys
import torch
import numpy as np
import torch.nn as nn
from .layers import main as l_main
from traits.api import Range, Enum, Int
from .gradient_extraction import Gradients
from torch.utils.data import Dataset, DataLoader
from traits.api import HasTraits, Instance, observe
from traitsui.api import Item, View, Group, HSplit, VSplit, CancelButton, OKButton
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor


def _init_weights_uniform(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.uniform_(m.weight)


def _init_weights_normal(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight)


def _init_weights_xavier_uniform(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)


def _init_weights_xavier_normal(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)


class FeedForwardNet(nn.Module):
    def __init__(self, layers):
        super(FeedForwardNet, self).__init__()
        net = []
        for i in range(len(layers) - 1):
            layer = nn.Linear(layers[i], layers[i + 1])
            net.append(layer)
            if i != len(layers) - 2:
                net.append(nn.ReLU())
        self.model = nn.Sequential(*net)
        self.fc = nn.Linear(layers[-1], 2)

    def forward(self, x):
        out = self.model(x)
        return self.fc(out)


class CustomDataset(Dataset):
    def __init__(self):
        self.X = torch.rand(1000, 4)
        self.y = torch.rand(1000, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Visualize(HasTraits):
    """
    This class creates the UI for gradient visualization using traits UI and mayavi
    """
    scene1 = Instance(MlabSceneModel, args=())

    learning_rate = Range(1e-5, 1e-3, value=1e-3)
    optimizer_type = Enum(
        'Adam',
        'SGD',
        'RMSprop'
    )
    lr_scheduler = Enum(
        'Exponential',
        'Step',
        'None'
    )
    param_init = Enum(
        'None',
        'Xavier Normal',
        'Xavier Uniform',
        'Normal',
        'Uniform'
    )
    batch_size = Int(16)

    height = 800
    width = 1200

    view = View(
        Group(
            Group(
                Item(name='scene1',
                     editor=SceneEditor(scene_class=MayaviScene),
                     label='Vector Cut Plane',
                     show_label=False,
                     resizable=True,
                     height=height, width=width),
            ),
            HSplit(
                Group(
                    VSplit(
                        Group(
                            Item(
                                name='learning_rate',
                                label='Learning Rate'
                            ),
                            Item(
                                name='optimizer_type',
                                label='Optimizer type'
                            ),
                            Item(
                                name='lr_scheduler',
                                label='Learning Rate Scheduler'
                            ),
                        ),
                        label='Optimizer Attributes',
                    ),
                    show_border=True,
                ),

                Group(
                    VSplit(
                        Group(
                            Item(
                                name='param_init',
                                label='Parameter Initialization Function'
                            ),
                        ),
                        label='Model Attributes',
                    ),
                    show_border=True,
                ),

                Group(
                    VSplit(
                        Group(
                            Item(
                                name='batch_size',
                                label='Batch Size'
                            ),
                        ),
                        label='Dataloader Attributes',
                    ),
                    show_border=True,
                ),
            ),
        ),
        buttons=[OKButton, CancelButton],
        resizable=True,
    )

    def __init__(self, model, optimizer, dataloader, scheduler=None):
        super(HasTraits, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.scheduler = scheduler
        self.gradient_data = Gradients()
        self.loss = None

    def update_arguments(self, model, optimizer, dataloader, loss=None, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.scheduler = scheduler
        self.loss = loss

    @observe('scene1.activated')
    def generate_vcp(self, event=None):
        fig = self.scene1.mlab.clf()
        gradients = self.gradient_data.extract_ff_data(self.model, self.loss)
        l_main(fig, gradients)

        self.scene1.mlab.view(azimuth=-137.34807145774826, elevation=68.99068625108039,
                              distance=9.20540886776696,
                              focalpoint=np.array([2.07276375, 1.5, 0.27631931]))

    @observe('optimizer_type, learning_rate, lr_scheduler')
    def send_optimizer_type(self, event=None):
        self.update_optimizer(self.learning_rate, self.lr_scheduler, self.optimizer_type)

    @observe('param_init')
    def send_param_init(self, event=None):
        self.update_param_init(self.param_init)

    @observe('batch_size')
    def send_batch_size(self, event=None):
        self.update_batch_size(self.batch_size)

    def update_optimizer(self, lr, lrs, optim_type):
        # Change Optimizer type
        if optim_type == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif optim_type == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        elif optim_type == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr)
        else:
            print(f"Given optimizer {optim_type} not supported. Use one of 'Adam', 'SGD', 'RMSprop'", file=sys.stderr)

        # Change Learning Rate
        for g in self.optimizer.param_groups:
            g['lr'] = lr

        # Change Learning Rate Scheduler
        if lrs:
            print("Make sure to use 'scheduler.step()' in pytorch code")
            if lrs == 'Exponential':
                print("Using gamma=0.9 for Exponential LR Scheduler")
                self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
            elif lrs == 'Step':
                print("Using step_size=10 for Step LR Scheduler")
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10)
            else:
                print(f"Given schedule {lrs} not supported. Use one of 'Exponential', 'Step'", file=sys.stderr)

        return

    def update_param_init(self, param_init: str):
        """
        Handler to apply the parameter initialization function on the linear layers of the model
        :param param_init: Type of parameter initialization. Options: 'Uniform', 'Normal', 'Xavier Uniform', 'Xavier Normal'
        :return:
        """
        if param_init:
            if param_init == "Uniform":
                self.model.apply(_init_weights_uniform)
            elif param_init == "Normal":
                self.model.apply(_init_weights_normal)
            elif param_init == "Xavier Uniform":
                self.model.apply(_init_weights_xavier_uniform)
            elif param_init == "Xavier Normal":
                self.model.apply(_init_weights_xavier_normal)
            else:
                print(f"Given parameter initialization {param_init}, which is not supported. Please give one of "
                      f"these: 'Uniform', 'Normal', 'Xavier Uniform', 'Xavier Normal'", file=sys.stderr)

    def update_batch_size(self, batch_size: int):
        """
        Updates the batchsize of the dataloader
        :param batch_size: new batchsize to be used
        :return:
        """
        if self.dataloader.sampler:
            self.dataloader = DataLoader(
                dataset=self.dataloader.dataset,
                batch_size=batch_size,
                num_workers=self.dataloader.num_workers,
                pin_memory=self.dataloader.pin_memory,
                drop_last=self.dataloader.drop_last,
                timeout=self.dataloader.timeout,
                sampler=self.dataloader.sampler,
                prefetch_factor=self.dataloader.prefetch_factor
            )
        else:
            self.dataloader = DataLoader(
                dataset=self.dataloader.dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=self.dataloader.num_workers,
                pin_memory=self.dataloader.pin_memory,
                drop_last=self.dataloader.drop_last,
                timeout=self.dataloader.timeout,
                prefetch_factor=self.dataloader.prefetch_factor
            )


def test_viz():
    custom = CustomDataset()
    dataloader = DataLoader(custom, batch_size=4,
                            shuffle=True, num_workers=0)
    objective = nn.MSELoss()
    model = FeedForwardNet([4, 5, 3, 2])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scene = Visualize(model, optimizer, dataloader)

    model.train()
    model.zero_grad()

    sample = torch.rand((8, 4))
    label = torch.randint(0, 2, (8, 2)).float()

    out = model(sample)

    loss = objective(out, label)
    loss.backward()

    scene.update_arguments(model, optimizer, dataloader, loss)

    print("Before")
    print(optimizer.defaults['lr'])
    print(optimizer)
    print(dataloader.batch_size)

    # scene = Visualize()
    scene.configure_traits()

    print("After")
    print(scene.optimizer.defaults['lr'])
    print(scene.optimizer)
    print(scene.dataloader.batch_size)
    return


# class FeedForwardNet(nn.Module):
#     def __init__(self, layers):
#         super(FeedForwardNet, self).__init__()
#         net = []
#         for i in range(len(layers) - 1):
#             layer = nn.Linear(layers[i], layers[i + 1])
#             net.append(layer)
#             if i != len(layers) - 2:
#                 net.append(nn.ReLU())
#         self.model = nn.Sequential(*net)
#         self.fc = nn.Linear(layers[-1], 2)
#
#     def forward(self, x):
#         out = self.model(x)
#         return self.fc(out)
#
#
# class CustomDataset(Dataset):
#     def __init__(self):
#         self.X = torch.rand(1000, 4)
#         self.y = torch.rand(1000, 1)
#
#     def __len__(self):
#         return len(self.y)
#
#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]
#
#
# def test_viz():
#     custom = CustomDataset()
#     dataloader = DataLoader(custom, batch_size=4,
#                             shuffle=True, num_workers=0)
#     objective = nn.MSELoss()
#     model = FeedForwardNet([4, 5, 3, 2])
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     scene = Visualize(model, optimizer, dataloader)
#
#     model.train()
#     model.zero_grad()
#
#     sample = torch.rand((8, 4))
#     label = torch.randint(0, 2, (8, 2)).float()
#
#     out = model(sample)
#
#     loss = objective(out, label)
#     loss.backward()
#
#     scene.update_arguments(model, optimizer, dataloader, loss)
#
#     print("Before")
#     print(optimizer.defaults['lr'])
#     print(optimizer)
#     print(dataloader.batch_size)
#
#     # scene = Visualize()
#     scene.configure_traits()
#
#     print("After")
#     print(scene.optimizer.defaults['lr'])
#     print(scene.optimizer)
#     print(scene.dataloader.batch_size)
