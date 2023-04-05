import copy
import os

import keras
import numpy as np
import pandas as pd
import shap
import tensorflow
import xgboost as xgb

from handler import get_data
from model_handler import api_model, XGBS_PARAMS

print(keras.__version__)
print(tensorflow.__version__)


def train_xgb_models():
    for org_name in ["worm", "cow", "human", "mouse"]:
        x, y = get_data(org_name, "train")
        train_data = xgb.DMatrix(x, y)
        org_model = xgb.train(params=XGBS_PARAMS, dtrain=train_data)
        org_model.save_model(f'models/xgb_{org_name}.dat')
        print(f"---{org_name} for XGB model has been saved---")


def calc_xgb_model_shap():
    org_list = ["worm", "mouse", "cow", "human"]
    for org_name in org_list:
        base_model = xgb.XGBClassifier(kwargs=XGBS_PARAMS)
        base_model.load_model(f'models/xgb_{org_name}.dat')
        copy_org_list = copy.deepcopy(org_list)
        copy_org_list.remove(org_name)
        for c_org_name in copy_org_list:
            print(f"---Starting src {org_name} and dst {c_org_name}---")
            x, y = get_data(c_org_name, "test")
            train_data = xgb.DMatrix(x[:500], y[:500])
            c_base_model = xgb.train(params=XGBS_PARAMS, dtrain=train_data)
            base_explainer = shap.TreeExplainer(base_model, x.values[:100])
            base_shap_values = base_explainer.shap_values(x.values)
            c_explainer = shap.TreeExplainer(c_base_model, x.values[:100])
            c_shap_values = c_explainer.shap_values(x.values)
            sub_res = np.absolute(np.subtract(c_shap_values, base_shap_values)).sum(axis=0)
            df = pd.DataFrame({"value": sub_res, "name": x.columns}).sort_values("value", ascending=False)
            df.to_csv(os.path.join("results", f"xgb_{org_name}_{c_org_name}.csv"), index=False)
        i = 9


def calc_xgb_model_base_shap():
    org_list = ["worm", "mouse", "cow", "human"]
    for org_name in org_list:
        base_model = xgb.XGBClassifier(kwargs=XGBS_PARAMS)
        base_model.load_model(f'models/xgb_{org_name}.dat')
        x, y = get_data(org_name, "test")
        base_explainer = shap.TreeExplainer(base_model, x.values[:100])
        base_shap_values = base_explainer.shap_values(x.values)
        sub_res = np.absolute(base_shap_values).sum(axis=0)
        df = pd.DataFrame({"value": sub_res, "name": x.columns}).sort_values("value", ascending=False)
        df.to_csv(os.path.join("results", f"xgb_{org_name}_{org_name}.csv"), index=False)


def fuck_off():
    org_list = ["worm", "mouse", "cow", "human"]
    for org_name in org_list:
        copy_org_list = copy.deepcopy(org_list)
        copy_org_list.remove(org_name)
        for c_org_name in copy_org_list:
            print(f"---Starting src {org_name} and dst {c_org_name}---")
            x, y = get_data(c_org_name, "test")
            base_model = xgb.XGBClassifier(kwargs=XGBS_PARAMS)
            base_model.load_model(f'models/xgb_{org_name}.dat')
            c_base_model = xgb.XGBClassifier(kwargs=XGBS_PARAMS)
            c_base_model.load_model(f'models/xgb_{c_org_name}.dat')
            base_explainer = shap.TreeExplainer(base_model, x.values[:100])
            base_shap_values = base_explainer.shap_values(x.values)
            c_explainer = shap.TreeExplainer(c_base_model, x.values[:100])
            c_shap_values = c_explainer.shap_values(x.values)
            sub_res = np.absolute(np.subtract(c_shap_values, base_shap_values)).sum(axis=0)
            df = pd.DataFrame({"value": sub_res, "name": x.columns}).sort_values("value", ascending=False)
            df.to_csv(os.path.join("results", f"xgb_{org_name}_{c_org_name}.csv"), index=False)
        i = 9


def train_models():
    for org_name in ["worm", "cow", "human", "mouse"]:
        x, y = get_data(org_name, "train")
        org_model = api_model(490)
        org_model.fit(x, y, epochs=1, verbose=0)
        org_model.save(f'models/ann_{org_name}')


def calc_model_shap():
    org_list = ["worm", "cow", "human", "mouse"]
    for org_name in org_list:
        base_model = keras.models.load_model(f'models/ann_{org_name}')
        copy_org_list = copy.deepcopy(org_list)
        copy_org_list.remove(org_name)
        for c_org_name in copy_org_list:
            print(f"---Starting src {org_name} and dst {c_org_name}---")
            x, y = get_data(c_org_name, "test")
            c_base_model = keras.models.load_model(f'models/ann_{org_name}')
            c_base_model.fit(x[0:500], y[0:500], epochs=10, verbose=0)
            base_explainer = shap.DeepExplainer(base_model, x.values[:100])
            base_shap_values = base_explainer.shap_values(x.values)
            c_explainer = shap.DeepExplainer(c_base_model, x.values[:100])
            c_shap_values = c_explainer.shap_values(x.values)
            sub_res = np.absolute(np.subtract(c_shap_values[0], base_shap_values[0])).sum(axis=0)
            df = pd.DataFrame({"value": sub_res, "name": x.columns}).sort_values("value", ascending=False)
            df.to_csv(os.path.join("results", f"ann_{org_name}_{c_org_name}.csv"), index=False)
        i = 9


def calc_ann_model_base_shap():
    org_list = ["worm", "cow", "human", "mouse"]
    for org_name in org_list:
        base_model = keras.models.load_model(f'models/ann_{org_name}')
        x, y = get_data(org_name, "test")
        base_explainer = shap.DeepExplainer(base_model, x.values[:100])
        base_shap_values = base_explainer.shap_values(x.values)
        sub_res = np.absolute(base_shap_values[0]).sum(axis=0)
        df = pd.DataFrame({"value": sub_res, "name": x.columns}).sort_values("value", ascending=False)
        df.to_csv(os.path.join("results", f"ann_{org_name}_{org_name}.csv"), index=False)


if __name__ == '__main__':
    calc_xgb_model_base_shap()
    calc_ann_model_base_shap()
    # fuck_off()
    # train_xgb_models()
    # calc_xgb_model_shap()
    # train_models()
    # calc_model_shap()
