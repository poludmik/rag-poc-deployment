# # Load data
import pickle

import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# iris_x, iris_y = datasets.load_iris(return_X_y=True)

# # Split iris data in train and test data
# # A random permutation, to split the data randomly
# np.random.seed(0)
# indices = np.random.permutation(len(iris_x))
# iris_x_train = iris_x[indices[:-10]]
# iris_y_train = iris_y[indices[:-10]]
# iris_x_test = iris_x[indices[-10:]]
# iris_y_test = iris_y[indices[-10:]]

# # Create and fit a nearest-neighbor classifier

# knn = KNeighborsClassifier()
# knn.fit(iris_x_train, iris_y_train)
# knn.predict(iris_x_test)

# # save model

# with open("model.pkl", "wb") as file:
#     pickle.dump(knn, file)


from google.cloud import storage

def read_weights(bucket_name, blob_name):
    """Read a blob from GCS using file-like IO"""

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # load the model from "model.pkl" on GCS
    with blob.open("rb") as file:
        model = pickle.load(file)

    return model

print(read_weights("bucket-temus-test-case", "model.pkl"))
model = read_weights("bucket-temus-test-case", "model.pkl")
print(model.predict([[1, 2, 3, 4]]))
