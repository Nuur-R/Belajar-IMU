import numpy as np
from micromlgen import port
from sklearn.ensemble import RandomForestClassifier
from glob import glob
from os.path import basename

def load_features(folder):
    dataset = None
    classmap = {}
    for class_idx, filename in enumerate(glob('%s/*.csv' % folder)):
        class_name = basename(filename)[:-4]
        classmap[class_idx] = class_name
        samples = np.loadtxt(filename, dtype=float, delimiter=';')
        labels = np.ones((len(samples), 1)) * class_idx
        samples = np.hstack((samples, labels))
        dataset = samples if dataset is None else np.vstack((dataset, samples))
    return dataset, classmap

features, classmap = load_features('dataset')
X, y = features[:, :-1], features[:, -1]
classifier = RandomForestClassifier(n_estimators=10, max_depth=10).fit(X, y)
c_code = port(classifier, classmap=classmap)
print(c_code)

# def get_classifier(features):
#     X, y = features[:, :-1], features[:, -1]
# 
#     return RandomForestClassifier(20, max_depth=10).fit(X, y)
# 
# if __name__ == '__main__':
#     features, classmap = load_features('/content/dataset')
#     classifier = get_classifier(features)
#     c_code = port(classifier, classmap=classmap)
#     print(c_code)