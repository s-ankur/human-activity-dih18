from vis.losses import ActivationMaximization
from vis.regularizers import TotalVariation, LPNorm

from model3d import model

# The name of the layer we want to visualize
# (see model definition in vggnet.py)
layer_name = 'dense_2'
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

output_class = [20]

losses = [
    (ActivationMaximization(layer_dict[layer_name], output_class), 2),
    (LPNorm(model.input), 10),
    (TotalVariation(model.input), 10)
]
# opt = Optimizer(model.input, losses)
# opt.minimize(max_iter=500, verbose=True, callbacks=[GifGenerator('opt_progress')])
