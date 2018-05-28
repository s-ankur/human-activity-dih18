from config import *
from cnn2d import cnn2d_model
from cnn3d import cnn3d_model
categories=glob.glob(os.path.join(IMAGE_PATH,'*'))

model=cnn2d_model(input_shape=SIZE+(3,),num_classes=len(categories))
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

def load_model():
    model.load_weights(MODEL_NAME)
    print ('Loaded model successfully')

def save_model():
    model.save_weights(MODEL_NAME)
    print ('Saved model successfully')
    

try:
    load_model()
except:
    pass
    
