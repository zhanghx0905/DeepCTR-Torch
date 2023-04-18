import argparse
import os

import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr_torch.inputs import DenseFeat, SparseFeat, get_feature_names
from deepctr_torch.models import *


def seed_everything(seed_value):
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense_embedding", action="store_true")
    parser.add_argument("--use_sample_data", action="store_true")
    parser.add_argument("--not_use_cin", action="store_true")
    parser.add_argument("--not_use_dnn", action="store_true")
    parser.add_argument("--save_path", type=str, default="ckpt/raw.pth")
    return parser.parse_args()


seed_everything(seed_value=17)
os.makedirs("ckpt", exist_ok=True)
DATA_FILE = "./criteo/train.txt"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SPARSE_FEATURES = ["C" + str(i) for i in range(1, 27)]
DENSE_FEATURES = ["I" + str(i) for i in range(1, 14)]


def load_data(use_sample=False):
    if use_sample:
        return pd.read_csv("./examples/criteo_sample.txt")
    else:
        n_used_sample = int(45840617)
        data = pd.read_csv(
            DATA_FILE,
            sep="\t",
            header=None,
            names=["label"] + DENSE_FEATURES + SPARSE_FEATURES,
            nrows=n_used_sample,
        )
        print(len(data))
        return data


def preprocessing(data):
    """Handle missing features(NaN),
    fill sparse features with -1, and dense features with 0

    Preprocess the features of the raw data
    Encode sparse features into integers between 0 and N-1,
    where N is the total number of possible values for the sparse feature
    Normalize dense features to range from 0 to 1."""
    data[SPARSE_FEATURES] = data[SPARSE_FEATURES].fillna("-1")
    data[DENSE_FEATURES] = data[DENSE_FEATURES].fillna(0)

    for feat in SPARSE_FEATURES:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[DENSE_FEATURES] = mms.fit_transform(data[DENSE_FEATURES])


def train(args, data):
    target = ["label"]

    """ Count the number of unique features for each sparse field 
    and record the name of the dense feature field.
    
    Build feature columns
    For sparse features: record the name of each sparse feature, 
        the number of possible values (vocabulary_size), 
        and the corresponding embedding dimension
    For dense features: record the name and dimension (default is 1)
    """
    fixlen_feature_columns = [
        SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=10)
        for feat in SPARSE_FEATURES
    ] + [DenseFeat(feat, 1) for feat in DENSE_FEATURES]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # Generate input data for model
    train, test = train_test_split(data, test_size=0.2, random_state=2022)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # Define Model, train, predict and evaluate
    dnn_hidden_units = (400, 400) if not args.not_use_dnn else ()
    cin_layer_size = (200, 200, 200) if not args.not_use_cin else ()
    model = xDeepFM(
        linear_feature_columns=linear_feature_columns,
        dnn_feature_columns=dnn_feature_columns,
        dnn_hidden_units=dnn_hidden_units,
        cin_layer_size=cin_layer_size,
        task="binary",
        l2_reg_embedding=1e-5,
        device=DEVICE,
        use_dense_embedding=args.dense_embedding,
    )
    model.compile("adam", "binary_crossentropy", metrics=["binary_crossentropy", "auc"])
    model_path = args.save_path
    print("save model in ", model_path)
    _history = model.fit(
        train_model_input,
        train[target].values,
        batch_size=1024,
        epochs=5,
        verbose=2,
        validation_split=0.2,
        model_path=model_path,
    )
    model.load_state_dict(torch.load(model_path))

    pred_ans = model.predict(test_model_input, 256)
    print("")
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))


if __name__ == "__main__":
    args = parse_args()
    print(args)
    data = load_data(args.use_sample_data)
    preprocessing(data)
    train(args, data)
