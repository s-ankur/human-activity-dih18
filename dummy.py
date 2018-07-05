from sklearn.dummy import DummyClassifier


class DummyClassifierAdaptor(DummyClassifier):

    def __init__(self, *_):
        super().__init__(strategy='most_frequent')

    def fit(self, X, y, *_):
        super().fit(X, y)

        class History:
            history = {}

        return History()


cnn2d_model = cnn3d_model = DummyClassifierAdaptor
