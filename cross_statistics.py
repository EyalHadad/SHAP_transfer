import copy
import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from inter_statistics import calc_by_metric


def write_cross_results(res_dict, metric, draw_heatmap):
    org_list = ["cow", "human", "mouse", "worm"]
    new_df = pd.DataFrame(columns=org_list, index=org_list)
    for k, v in res_dict.items():
        r = k.split("#")[0]
        c = k.split("#")[1]
        new_df[r][c] = v
    if draw_heatmap:
        new_df = new_df.astype(float)
        sns.clustermap(new_df, cmap="Blues")
        plt.savefig(os.path.join("metrics_results/", f"cross_{metric}.png"))
    else:
        new_df.to_csv(os.path.join("metrics_results/", f"cross_{metric}.csv"))


def calc_cross_statistic(_metric, draw_heatmap=True):
    res_dict = {}
    org_list = ["worm", "cow", "human", "mouse"]
    for org_name in org_list:
        copy_org_list = copy.deepcopy(org_list)
        for c_org_name in copy_org_list:
            print(f"---Starting src {org_name} and dst {c_org_name}---")
            ann_df_src = pd.read_csv(os.path.join("results", f"ann_{org_name}_{c_org_name}.csv"))
            xgb_df_src = pd.read_csv(os.path.join("results", f"xgb_{org_name}_{c_org_name}.csv"))
            res_dict[f"{org_name}#{c_org_name}"] = calc_by_metric(ann_df_src, xgb_df_src, _metric)
    write_cross_results(res_dict, _metric, draw_heatmap)


if __name__ == '__main__':
    for _metric in ["WILCOXON10", "WILCOXON20", "SPEARMANR", "JACARD10", "JACARD20"]:
        print(f"--- Starting {_metric} ---")
        calc_cross_statistic(_metric, True)
