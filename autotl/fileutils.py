import os
import tempfile

try:
    import cPickle as pickle
except:
    import pickle as pickle


def temp_folder_generator():
    sys_temp = tempfile.gettempdir()
    path = os.path.join(sys_temp, 'autokeras')
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def has_file(path):
    return os.path.exists(path)


def pickle_from_file(path):
    return pickle.load(open(path, 'rb'))


def pickle_to_file(obj, path):
    return pickle.dump(obj, open(path, 'wb'))


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
