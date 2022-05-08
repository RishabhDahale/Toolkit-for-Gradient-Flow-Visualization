This toolkit enables the vizualization of gradients. This can be used as a debugging strategy to better understand the 
network and take decisions to update the hyperparameters of the model.

```text
Note: This work is done as a part of AE6102 course project at IIT Bombay.
```

To use this package you will need to install Mayavi from [this link](https://docs.enthought.com/mayavi/mayavi/installation.html)

## Installation

1. Clone this repo with the following command
```bash
git clone https://github.com/RishabhDahale/Toolkit-for-Gradient-Flow-Visualization.git
cd Toolkit-for-Gradient-Flow-Visualization
```
2. Use the following commands in the cloned repo

Make sure you have updated the pip before installing. You can use the following command to update pip

```bash
python -m pip install --upgrade pip
```

### Installing the package using pip
To install the package using pip use the following command in the project folder

```bash
pip install .
```

### Testing

You should be able to import with the following command:
```python
>>> from grad_viz import visualizer
>>> visualizer.Visualize
<class 'grad_viz.visualizer.Visualize'>
```
