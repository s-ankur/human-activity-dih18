from cnn2d import * 
from config import *

model=cnn_model(input_shape=SIZE+(3,),num_classes=2)
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
    
