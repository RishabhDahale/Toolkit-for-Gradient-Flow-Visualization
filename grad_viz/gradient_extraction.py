import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import OrderedDict


class Gradients:
    def __init__(self, name='FeedForward'):
        self.name = name

    def extract_layer_data(self, model, loss, layer: str):
        """
        Returns the gradients of layer 'layer'
        :param model: pytorch model under consideration
        :param loss: loss at the specific iteration
        :param layer: name of the layer of the model of which to extract the gradients
        :return: Returns a dict mapping layers to the corresponding weights and averaged gradients
        """
        data = OrderedDict()
        data['metadata'] = {}
        for i, (n, p) in enumerate(model.named_parameters()):
            name = n.split('.')
            n = '.'.join(name[:-1])
            if n==layer and p.requires_grad:
                if n not in data:
                    data[n] = {'weight': {}, 'bias': {}}

                data[n][name[-1]]['parameters'] = p.cpu().detach().numpy()
                data[n][name[-1]]['gradients'] = p.grad.abs().numpy()

        data['metadata']['loss'] = loss.item()
        return data

    def extract_ff_data(self, model, loss):
        """
        Returns the gradients for all the layers of ff model
        :param model: pytorch model under consideration
        :param loss: loss at the specific iteration
        :return: returns a dict mapping layers to the corresponding weights and averages gradients of the FF layers
        """
        data = OrderedDict()
        data['metadata'] = {}
        architecture = []
        for i, (n, p) in enumerate(model.named_parameters()):
            name = n.split('.')
            n = '.'.join(name[:-1])
            
            if p.requires_grad:
                if i==0:
                    architecture.append(p.size(1))

                if n not in data:
                    data[n] = {'weight': {}, 'bias': {}}

                data[n][name[-1]]['parameters'] = p.cpu().detach().numpy()
                if name[-1]=='weight':
                    grads = p.grad.abs().mean(0).numpy()
                    data[n][name[-1]]['gradients'] = grads
                    architecture.append(grads.size)
                elif name[-1]=='bias':
                    data[n][name[-1]]['gradients'] = p.grad.abs().numpy()

        data['metadata']['architecture'] = architecture
        data['metadata']['loss'] = loss.item()
        return data

    def extract_pdf_data(self, model, loss):
        """
        Used for the layers other than feed forward layer. Generated and returns a pdf for these layers
        :param model: pytorch model under consideration
        :param loss: loss at the specific iteration
        :return: returns a dict mapping layers to the corresponding weights and pdf of these gradients
        """
        data = OrderedDict()
        data['metadata'] = {}
        for i, (n, p) in enumerate(model.named_parameters()):
            name = n.split('.')
            n = '.'.join(name[:-1])

            if p.requires_grad:
                if n not in data:
                    data[n] = {'weight': {}, 'bias': {}}

                data[n][name[-1]]['parameters'] = p.cpu().detach().numpy()
                data[n][name[-1]]['gradients'] = p.grad.abs().numpy()

        data['metadata']['loss'] = loss.item()
        return data


if __name__ == "__main__":

    class FeedForwardNet(nn.Module):
        def __init__(self, layers):
            super(FeedForwardNet, self).__init__()
            net = []
            for i in range(len(layers)-1):
                layer = nn.Linear(layers[i], layers[i+1])
                net.append(layer)
                if i != len(layers)-2:
                    net.append(nn.ReLU())
            self.model = nn.Sequential(*net)
            self.fc = nn.Linear(layers[-1], 2)

        def forward(self, x):
            out = self.model(x)
            return self.fc(out)

    gradient_extractor = Gradients()

    objective = nn.MSELoss()

    model = FeedForwardNet([4, 5, 3, 2])
    model.train()
    model.zero_grad()

    sample = torch.rand((8, 4))
    label = torch.randint(0, 2, (8, 2)).float()

    out = model(sample)

    loss = objective(out, label)
    loss.backward()

    data = gradient_extractor.extract_pdf_data(model, loss)

    x = data['model.2']['weight']['gradients'].reshape(-1)

    #visualize KDE
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x_eval = np.linspace(-.2, .2, num=200)
    ax.plot(x_eval, kde(x_eval), 'k-')

