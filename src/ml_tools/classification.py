from sklearn.preprocessing import LabelEncoder, OneHotEncoder


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
