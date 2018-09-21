from vis.losses import ActivationMaximization
from vis.regularizers import TotalVariation, LPNorm
from vis.optimizer import Optimizer
from model import model

from keras import activations
layer_idx =-1
model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)
losses = [
    (ActivationMaximization(model.layers[-2], 3), 2),
    (LPNorm(model.input), 10),
    (TotalVariation(model.input), 10)
]
opt = Optimizer(model.input, losses)
opt.minimize(max_iter=500, verbose=True, callbacks=[GifGenerator('opt_progress')])

