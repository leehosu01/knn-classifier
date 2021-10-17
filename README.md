# knn-classifier
cosine similarity based exponential distance weighted classifier


### usage
---------------
```python
from knn.torch import predict as knn_predict
from knn.numpy import predict as knn_predict
```
```python
from tensorflow.keras import datasets
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_labels = train_labels.reshape((-1, ))
test_labels = test_labels.reshape((-1, ))

from knn.torch import predict as knn_predict
%time (knn_predict(test_images.astype("float32"), train_images.astype("float32"), train_labels, batchsize = 512, device = 'cuda').cpu().numpy() == test_labels).mean()

from knn.numpy import predict as knn_predict
%time (knn_predict(test_images.astype("float32"), train_images.astype("float32"), train_labels, batchsize = 512) == test_labels).mean()
```