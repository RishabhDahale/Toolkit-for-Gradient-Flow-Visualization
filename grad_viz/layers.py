import numpy as np
from mayavi import mlab

TEXT_SCALE = 0.1
POINT_SCALE = 0.25


class Layer:
    def __init__(self, neurons, x_offset, name, z_plane=0):
        """
        Layer class. Contains all the information and functions needed for
        plot visualization for MLP
        :param neurons: number of neurons
        :param x_offset: x_offset of the layer
        :param name: name to be given to the layer
        :param z_plane: offset to the given in the z-plane  (Optional)
        """
        self.neurons = neurons
        self.x_offset = x_offset
        self.z_plane = z_plane
        self.name = name
        self.points = None
        self.y = None
        self.grad_viz = None
    
    def get_locations(self):
        """
        Returns the x, y, z coordinates of the neurons
        :return: list of n tuple (x, y, z)
        """
        assert self.y is not None, f"Layer: {self.name} : mlab_points not called. Call it first as y not initialized"
        x1 = np.mgrid[self.x_offset:self.x_offset+self.neurons:self.neurons*1j]
        points = []
        for x_coor in x1:
            points.append((x_coor, self.y, self.z_plane))
        return points
    
    def mlab_points(self, y=None):
        if self.points is None:
            if y is None:
                y = self.y
                assert y is not None, 'Layer location on y-axis not given'
            else:
                self.y = y
            self.points = np.mgrid[
                            self.x_offset:self.neurons+self.x_offset:self.neurons*1j,
                            y:y:1j,
                            self.z_plane:self.z_plane:1j
                          ]
        return self.points
    
    def add_text(self, fig):
        """
        Adds the name of the layers to the figure 'fig'
        :param fig: figure to which text needs to be added
        :return:
        """
        mlab.text3d(
            -1,
            self.y,
            self.z_plane,
            text=self.name,
            figure=fig,
            scale=TEXT_SCALE
        )
    
    def add_layer(self, fig):
        """
        Add the layers as the 3D points in the figure 'fig'
        :param fig: figure to which neurons needs to be added
        :return:
        """
        mlab.points3d(
            *self.points,
            figure=fig,
            scale_factor=POINT_SCALE
        )
    
    def add_to_plot(self, fig, y, gradient_values):
        """
        Generated and adds the neurons, layer name and the gradients
        :param fig: figure to which gradient needs to be added
        :param y: y location of the layer
        :param gradient_values: gradient values of the neurons
        :return:
        """
        self.mlab_points(y)
        self.add_layer(fig)
        self.add_text(fig)
        self.add_gradient(fig, gradient_values)
    
    def add_gradient(self, fig, gradient_values):
        """
        Adds the gradients as the quiver3D plot on top of the neurons
        :param fig: figure to which
        :param gradient_values: gradient of the neurons
        :return:
        """
        grads = np.array(gradient_values)
        grads = grads[..., np.newaxis, np.newaxis]
        # print(grads.shape, self.points[2].shape)
        self.grad_viz = mlab.quiver3d(
            *self.points,
            np.zeros_like(self.points[0]),
            np.zeros_like(self.points[1]),
            np.ones_like((self.points[2]))*grads,
            figure=fig,
            # scale_factor=POINT_SCALE,
            # mode='2dhooked_arrow',
            # mode='2dthick_arrow',
            # mode='cone',
        )
        # glyph1.glyph.glyph_source.glyph_source = glyph1.glyph.glyph_source.glyph_dict['arrow_source']
    
    def update_grad(self, grad):
        if self.grad_viz is not None:
            # delete self.grad_viz object
            pass
        self.grad_viz = None    # some mlab plot thing which adds the vectors to the neurons


class Network:
    def __init__(self, layers: list, fig):
        for layer in layers:
            assert isinstance(layer, Layer), 'Incorrect object given to the layers parameter'
        self.layers = [(y, layer) for y, layer in enumerate(layers)]
        self.fig = mlab.clf(figure=fig)
    
    def plot_layers(self, gradient_values):
        for i, (y, l) in enumerate(self.layers):
            l.add_to_plot(self.fig, y, gradient_values[i])
    
    def plot_interconnects(self):
        prev_coordinates = self.layers[0][1].get_locations()
        for _, layer in self.layers[1:]:
            curr_coordinates = layer.get_locations()
            for pPoint in prev_coordinates:
                for cPoint in curr_coordinates:
                    x = [pPoint[0], cPoint[0]]
                    y = [pPoint[1], cPoint[1]]
                    z = [pPoint[2], cPoint[2]]
                    mlab.plot3d(
                        x,
                        y,
                        z,
                        line_width=0.1,
                        tube_radius=0.005,
                        figure=self.fig,
                    )
            prev_coordinates = curr_coordinates


def main(fig, gradients):
    # layers = {"Layer 1": [4, 1],
    #           "Layer 2": [6, 0],
    #           "Layer 3": [3, 1.5]}
    # layers = {"Layer 1": [4, 0, 1],}

    architecture = gradients['metadata']['architecture']
    loss = gradients['metadata']['loss']

    max_len = max(architecture)
    
    layers = {}
    gradient_values = []
    for i, (k, v) in enumerate(gradients.items()):
        if k!='metadata':
            neurons = len(v['weight']['gradients'])
            layers[f'Layer {i}'] = [neurons, (max_len-neurons)/2]
            gradient_values.append(v['weight']['gradients'])

    print(layers)
    print(gradient_values)

    layers = [Layer(d[0], d[1], n) for n, d in layers.items()]
    net = Network(layers, fig)

    # gradient_values = [[1, 3, 5, 2], [1, 2, 3, 4, 5, 6], [3, 1, 2]]
    # gradients = [6, 13, 8, 12]
    # for grad in gradients:
    #     net.plot_layers(grad)
    net.plot_layers(gradient_values)
    net.plot_interconnects()
    # mlab.show()


if __name__=="__main__":
    main()