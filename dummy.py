from sklearn.dummy import DummyClassifier


class DummyClassifierAdaptor:

    def __init__(self, *_, **__):
        self.classifier = DummyClassifier(strategy='most_frequent')

    def fit(self, X, y, *_, **__):
        self.classifier.fit(X, y)

        class History:
            history = {}

        return History()

    def __getattr__(self, item):
        return getattr(self.classifier, item, lambda *_, **__: True)

    def to_json(self):
        return 'Dummy'


cnn2d_model = cnn3d_model = DummyClassifierAdaptor
