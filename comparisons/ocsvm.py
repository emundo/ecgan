import torch
from ecgan.utils.embeddings import assert_and_reshape_dim, calculate_pca

from ecgan.evaluation.metrics.classification import FScoreMetric, MCCMetric, AUROCMetric

from ecgan.training.trainer import Trainer

from ecgan.utils.custom_types import SplitMethods
from ecgan.utils.splitting import create_splits, load_split, select_channels

from ecgan.utils.miscellaneous import load_pickle_numpy, to_torch, to_numpy
from sklearn.svm import OneClassSVM
from sklearn.linear_model import SGDOneClassSVM
import os

LOAD_DIR='data'
DATASET='mitbih_beats'
target_dir = os.path.join(LOAD_DIR, DATASET, 'processed')

data = to_torch(load_pickle_numpy(os.path.join(target_dir, 'data.pkl')))
label = to_torch(load_pickle_numpy(os.path.join(target_dir, 'label.pkl')))
print(data.shape)

split_method = 'normal_only'

split_indices = create_splits(
    data,
    label,
    seed=42,
    folds=5,
    method=SplitMethods.NORMAL_ONLY,
    split=(0.85, 0.15),
)
train_x, test_x, vali_x, train_y, test_y, vali_y = load_split(
    data, label, index_dict=split_indices, fold=2
)


train_x = select_channels(train_x, 1)
vali_x = select_channels(vali_x, 1)
test_x = select_channels(test_x, 1)

train_y[train_y != 0] = 1
vali_y[vali_y != 0] = 1
test_y[test_y != 0] = 1

print(train_x.shape, train_y.shape, type(train_x), torch.unique(train_y, return_counts=True), torch.unique(test_y, return_counts=True))

# train_x=assert_and_reshape_dim(to_numpy(train_x))
# test_x=assert_and_reshape_dim(to_numpy(test_x))
# TEST

# Take first 50 PCA dims as in BeatGAN
train_x,_=calculate_pca(to_numpy(train_x),50)
test_x,_=calculate_pca(to_numpy(test_x),50)
print(train_x.shape)
train_x=to_torch(train_x)
test_x=to_torch(test_x)

for nu in [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
    clf = OneClassSVM(nu=nu, cache_size=3000).fit(train_x)
    pred=clf.predict(test_x)

    fscore = FScoreMetric().calculate(test_y, pred)
    mcc = MCCMetric().calculate(test_y, pred)
    auroc = AUROCMetric().calculate(test_y, pred)

    print("OC SVM (nu={}): Fscore: {}\n MCC: {} \n AUROC: {}".format(nu, fscore, mcc, auroc))

    clf = SGDOneClassSVM(nu=nu).fit(train_x)
    pred=clf.predict(test_x)

    fscore = FScoreMetric().calculate(test_y, pred)
    mcc = MCCMetric().calculate(test_y, pred)
    auroc = AUROCMetric().calculate(test_y, pred)

    print("OC SVM SGD (nu={}): Fscore: {}\n MCC: {} \n AUROC: {}".format(nu, fscore, mcc, auroc))