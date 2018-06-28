from keras.utils.vis_utils import plot_model
import sys



if  len(sys.argv)==2 and sys.argv[1]=='3d':
  from model3d import *
else:
  from model import *
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
