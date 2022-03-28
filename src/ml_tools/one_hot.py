from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np


def categorical_encode(values):  # *
    """Converts categories of any type to one-hot encoded numpy array.

    :param values: Collection of categorical data.
    :type values: iterable
    :return: One-hot encoded numpy array.
    :rtype: numpy.ndarray
    """
    
    # Integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    
    # Binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    return onehot_encoder.fit_transform(integer_encoded)

def one_hot_acc(y_true, y_pred):
    out = [np.array_equal(t, p) for t, p in zip(y_true, y_pred)]
    return sum(out)/len(out)
