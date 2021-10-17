import warnings
import numpy as np
def __predict(test_X:np.ndarray, train_X:np.ndarray, train_Y:np.ndarray, knn_k:int, knn_t:float, batchsize: int):
    assert len(test_X.shape) == 2   # [N, D]
    assert len(train_X.shape) == 2  # [D, M]
    assert len(train_Y.shape) == 1  # [M]
    assert train_X.shape[0] == test_X.shape[1]
    if train_X.shape[1] <= knn_k:
        warnings.warn(f"samples count = {train_X.shape[0]}, but consider nearest {knn_k} elements.  ", UserWarning)
    
    pred_label = []
    one_hot_class = np.eye(train_Y.max() + 1)[train_Y].astype(test_X.dtype) #[M, C]
    
    if test_X.shape[0] <= batchsize:
        splited_tensor = [test_X]
    else:
        slices_size = [batchsize] * (test_X.shape[0]//batchsize) 
        if test_X.shape[0]%batchsize != 0:
            slices_size += [test_X.shape[0]%batchsize]
        splited_tensor = np.split(test_X, slices_size, axis = 0)

    for part_of_test_X in splited_tensor:
        similiarity = part_of_test_X @ train_X  #[N, M]
        sim_weight = np.exp(similiarity / knn_t) #[N, M]
        sim_indices = similiarity.argsort(axis = -1)[:, -knn_k:] #[N, K]
        mask = np.zeros_like(similiarity)
        np.put_along_axis(mask, sim_indices, 1, axis = -1)
        pred_score = (sim_weight * mask) @ one_hot_class
        pred_label.append(pred_score.argmax(axis = -1))

    pred_label = np.concatenate(pred_label, axis = 0)
    return pred_label
def predict(test_X, train_X, train_Y, knn_k:int = 200, knn_t:float = 0.1, batchsize : int = 512):
    if type(test_X) is not np.ndarray:     test_X  = np.asarray(test_X)
    if type(train_X) is not np.ndarray:    train_X = np.asarray(train_X)
    if type(train_Y) is not np.ndarray:    train_Y = np.asarray(train_Y)
    assert "int" in str(train_Y.dtype)
    if len(train_X.shape) != len(test_X.shape):
        raise Exception(f"(rank(train_X) == {len(train_X.shape)}) != (rank(test_X) == {len(test_X.shape)})")
    if len(train_X.shape) > 2:
        warnings.warn(f"rank(train_X) = {len(train_X.shape)} > 2 . we reshape train_X ", UserWarning)
        train_X = train_X.reshape(train_X.shape[0], -1)
    if len(test_X.shape) > 2:
        warnings.warn(f"rank(test_X) = {len(test_X.shape)} > 2 . we reshape test_X ", UserWarning)
        test_X = test_X.reshape(test_X.shape[0], -1)
    test_X  = test_X / np.linalg.norm(test_X, ord=2, axis=-1, keepdims=True)
    train_X = train_X / np.linalg.norm(train_X, ord=2, axis=-1, keepdims=True)
    train_X = train_X.T
    return __predict(test_X, train_X, train_Y, knn_k, knn_t, batchsize = batchsize)