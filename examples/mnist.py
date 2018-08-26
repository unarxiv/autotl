import tensorlayer as tl
from autotl.tasks.cv.ImageSupervised import ImageSupervised

X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 784))
