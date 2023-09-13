import abc

class MlModel(metaclass=abc.ABCMeta):

    def __init__(self, path_to_weights):
        self.path_to_weights = path_to_weights

    @abc.abstractmethod
    def instantiate_model(self):
        pass

    @abc.abstractmethod
    def train_model(self):
        pass

    def predict(self, text):
        pass

