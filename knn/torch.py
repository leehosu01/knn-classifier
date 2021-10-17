import warnings
import torch
def __predict(test_X:torch.tensor, train_X:torch.tensor, train_Y:torch.tensor, knn_k:int, knn_t:float, batchsize: int):
    assert len(test_X.shape) == 2   # [N, D]
    assert len(train_X.shape) == 2  # [D, M]
    assert len(train_Y.shape) == 1  # [M]
    assert train_X.shape[0] == test_X.shape[1]
    if train_X.shape[1] <= knn_k:
        warnings.warn(f"samples count = {train_X.shape[0]}, but consider nearest {knn_k} elements.  ", UserWarning)
    
    pred_scores = []
    one_hot_class = torch.nn.functional.one_hot(train_Y).type(test_X.dtype)    #[M, C]
    
    if test_X.shape[0] <= batchsize:
        splited_tensor = [test_X]
    else:
        slices_size = [batchsize] * (test_X.shape[0]//batchsize) 
        if test_X.shape[0]%batchsize != 0:
            slices_size += [test_X.shape[0]%batchsize]
    splited_tensor = test_X.split(slices_size, dim = 0)

    for part_of_test_X in splited_tensor:
        similiarity = part_of_test_X @ train_X  #[N, M]
        sim_weight, sim_indices = similiarity.topk(k=knn_k, dim=-1, sorted = False) #[N, K], [N, K]
        sim_weight = (sim_weight / knn_t).exp() #[N, K]
        pred_score = torch.zeros_like(similiarity).scatter(-1, sim_indices, sim_weight) @ one_hot_class
        pred_scores.append(pred_score)

    pred_scores = torch.cat(pred_scores, dim = 0)
    #pred_score = sim_weight * one_hot_class.gather(dim = 0, index = sim_indices)   #[N, C]
    #pred_score = pred_score.sum(dim = -2) #[N, C]
    
    pred_label = pred_scores.argmax(dim = -1)
    return pred_label
def predict(test_X, train_X, train_Y, knn_k:int = 200, knn_t:float = 0.1, batchsize : int = 512, device = None):
    if not torch.is_tensor(test_X):     test_X  = torch.tensor(test_X)
    if not torch.is_tensor(train_X):    train_X = torch.tensor(train_X)
    if not torch.is_tensor(train_Y):    train_Y = torch.tensor(train_Y)
    assert train_Y.dtype in [torch.int, torch.int16, torch.int32, torch.int64]
    if len(train_X.shape) != len(test_X.shape):
        raise Exception(f"(rank(train_X) == {len(train_X.shape)}) != (rank(test_X) == {len(test_X.shape)})")
    if len(train_X.shape) > 2:
        warnings.warn(f"rank(train_X) = {len(train_X.shape)} > 2 . we reshape train_X ", UserWarning)
        train_X = train_X.reshape(train_X.shape[0], -1)
    if len(test_X.shape) > 2:
        warnings.warn(f"rank(test_X) = {len(test_X.shape)} > 2 . we reshape test_X ", UserWarning)
        test_X = test_X.reshape(test_X.shape[0], -1)
    test_X = torch.nn.functional.normalize(test_X, dim=-1)
    train_X = torch.nn.functional.normalize(train_X, dim=-1)
    train_X = train_X.T
    if device:
        test_X      = test_X.to(device)
        train_X     = train_X.to(device)
        train_Y     = train_Y.to(device)
    return __predict(test_X, train_X, train_Y, knn_k, knn_t, batchsize = batchsize)