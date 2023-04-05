import copy
import math
import os

import pandas as pd
from scipy import stats
from sklearn.metrics import jaccard_score

import seaborn as sns
from matplotlib import pyplot as plt


def get_ordered_top_features(df_src, df_trg, n_top=10):
    top_fetures = set(list(df_trg['name'][:n_top]) + list(df_src['name'][:n_top]))
    df_src = df_src[df_src['name'].isin(top_fetures)]
    df_trg = df_trg[df_trg['name'].isin(top_fetures)]
    merged_frame = pd.merge(df_src, df_trg, on="name", how='inner')
    return merged_frame


def calc_by_metric(df_src, df_trg, metric_name):
    res = None
    if metric_name == "JACARD10":
        res = jaccard_score(df_src['name'][:10], df_trg['name'][:10], average='weighted')
    if metric_name == "JACARD20":
        res = jaccard_score(df_src['name'][:20], df_trg['name'][:20], average='weighted')
    if metric_name == "SPEARMANR":
        normalized_src_df = normalize_vector(df_src)
        normalized_trg_df = normalize_vector(df_trg)
        new_df = get_ordered_top_features(normalized_src_df, normalized_trg_df, n_top=df_src.shape[0])
        res = stats.spearmanr(new_df['value_x'], new_df['value_y']).correlation
    if metric_name == "WILCOXON10":
        normalized_src_df = normalize_vector(df_src)
        normalized_trg_df = normalize_vector(df_trg)
        new_df = get_ordered_top_features(normalized_src_df, normalized_trg_df, n_top=10)
        res = stats.wilcoxon(new_df['value_x'], new_df['value_y']).pvalue
        if res > 0.005:
            res = math.log(res, 10) * -1
        else:
            res = 0

    if metric_name == "WILCOXON20":
        normalized_src_df = normalize_vector(df_src)
        normalized_trg_df = normalize_vector(df_trg)
        new_df = get_ordered_top_features(normalized_src_df, normalized_trg_df, n_top=20)
        res = stats.wilcoxon(new_df['value_x'], new_df['value_y']).pvalue
        if res > 0.005:
            res = math.log(res, 10) * -1
        else:
            res = 0
    return res


def normalize_vector(df_src):
    df_src["value"] = (df_src['value'] - df_src['value'].min()) / (df_src['value'].max() - df_src['value'].min())
    return df_src


def write_results(res_dict, method, metric, draw_heatmap):
    keys = set(x.split("#")[0].replace(f"{method}_", "") for x in res_dict.keys())
    row_order = sorted(keys, key=lambda item: (item.split("_")[0]))
    col_order = sorted(keys, key=lambda item: (item.split("_")[1]))
    new_df = pd.DataFrame(columns=col_order, index=row_order)
    for k, v in res_dict.items():
        r = k.split("#")[0].replace(f"{method}_", "")
        c = k.split("#")[1].replace(f"{method}_", "")
        new_df[r][c] = v
    df_max = max(new_df.max())
    if metric == "WILCOXON10" or metric == "WILCOXON20":
        new_df.fillna(df_max, inplace=True)
    else:
        new_df.fillna(1, inplace=True)
    if draw_heatmap:
        sns.clustermap(new_df, cmap="Blues", vmin=0, vmax=df_max)
        plt.savefig(os.path.join("metrics_results/", f"{method}_{metric}.png"))
    else:
        new_df.to_csv(os.path.join("metrics_results/", f"{method}_{metric}.csv"))


def calc_statistic(method="ann", metric="JACARD10", draw_heatmap=True):
    res_dict = {}
    s_list_files = [x for x in os.listdir("results/") if method in x]
    for org_n in ["cow", "worm", "human", "mouse"]:
        s_list_files.remove(f"{method}_{org_n}_{org_n}.csv")
    for src_file in s_list_files:
        df_src = pd.read_csv(os.path.join("results", src_file))
        d_list_files = copy.deepcopy(s_list_files)
        d_list_files.remove(src_file)
        for trg_file in d_list_files:
            df_trg = pd.read_csv(os.path.join("results", trg_file))
            res_dict[f"{src_file}#{trg_file}".replace(".csv", "")] = calc_by_metric(df_src, df_trg, metric)
    write_results(res_dict, method, metric, draw_heatmap)


if __name__ == '__main__':
    for _method in ["xgb", "ann"]:
        for _metric in ["WILCOXON10", "WILCOXON20", "SPEARMANR", "JACARD10", "JACARD20"]:
            print(f"--- Starting {_method}_{_metric} ---")
            calc_statistic(_method, _metric, True)
