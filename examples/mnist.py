import sys
sys.path.append('../')

import tensorlayer as tl
from autotl.tasks.cv.ImageSupervised import ImageClassifier

X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 784))

X_train = X_train.reshape(X_train.shape + (1,))
X_test = X_test.reshape(X_test.shape + (1,))

clf = ImageClassifier(verbose=True,augment=False)
clf.fit(X_train,y_train, time_limit=0.5 * 60 * 60)
clf.evaluate(X_test, y_test)