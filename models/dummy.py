from sklearn.dummy import DummyClassifier
from keras.utils import np_utils
import pickle

STRATEGY = 'uniform'


# uniform -> randomly using uniform distribution
# most_frequent -> always predicts the most frequent label in the training set
# stratified -> generates predictions by respecting the training set’s class distribution.
# constant -> always predicts a constant label that is provided by the user. expects a constant value in constructor

class DummyClassifierAdaptor:

    def __init__(self, input_shape, num_classes, *_, **__):
        self.classifier = DummyClassifier(strategy=STRATEGY)
        self.input_shape = input_shape
        self.num_classes = num_classes

    def fit(self, X, y, *_, **__):
        self.classifier.fit(X.reshape(X.shape[0], -1), y.argmax(1))

        class History:
            history = {}

        return History()

    def predict(self, X, *_, **__):
        y_pred = self.classifier.predict(X.reshape(X.shape[0], -1))
        y_pred = np_utils.to_categorical(y_pred, self.num_classes)
        return y_pred

    def save_weights(self, path, *_, **__):
        with open(path, 'wb') as model_file:
            pickle.dump(self.classifier, model_file)

    def load_weights(self, path, *_, **__):
        with open(path, 'rb') as model_file:
            self.classifier = pickle.load(model_file)

    def __getattr__(self, item):
        return getattr(self.classifier, item, lambda *_, **__: True)

    def to_json(self):
        return str(self.classifier.get_params())


dummy_model = DummyClassifierAdaptor
