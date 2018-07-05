from sklearn.dummy import DummyClassifier

cnn2d_model = cnn3d_model = lambda *args: DummyClassifier(strategy='most_frequent')
