import _pickle as cPickle


def load_model():
    with open('ts/my_classifier.pickle', 'rb') as file:
        model = cPickle.load(file)
    return model
