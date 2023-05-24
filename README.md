# SHAP_transfer
TransferSHAP is a Python package that provides a methodology for calculating feature importance 
in transfer learning tasks with tabular data. 
This repository contains two main files: `TransferSHAP.py` and `cross_statistics.py`.

## TransferSHAP.py
The `TransferSHAP.py` file includes two methods, one for deep learning models and one for XGBoost 
models, to calculate feature importance using the TransferSHAP technique.

### Methods
* `xgb_shap_transfer_features`: This function loads XGBoost models from the project folder for all 
source-target pairs and generates the transfer SHAP feature importance. 
The results are saved as a CSV file in the `results` folder.

* `ann_shap_transfer`: This function loads deep learning models from the project folder for all 
source-target pairs and generates the transfer SHAP feature importance. 
The results are saved as a CSV file in the `results` folder.

## cross_statistics.py
The `cross_statistics.py` file is used to calculate statistics based on the CSV output files generated
by the TransferSHAP method.

### Usage
The script takes the following inputs:

* `metric_name`: The name of the correlation metric to be calculated. The supported metrics are:
  * `"WILCOXON10"`: Wilcoxon correlation based on the top 10 features of both models
  * `"WILCOXON20"`: Wilcoxon correlation based on the top 20 features of both models
  * `"SPEARMANR"`: Spearman correlation between two transfer learning models
  * `"JACARD10"`: Jaccard similarity based on the top 10 features of both models
  * `"JACARD20"`: Jaccard similarity based on the top 20 features of both models
* `draw_heatmap`: A boolean indicating whether to draw a heatmap of the results (`True`) or save it as a CSV file (`False`).

The script generates the desired statistics based on the input and either displays the heatmap or saves the CSV file accordingly.

## Usage
To use the TransferSHAP package, follow these steps:

1. Clone this repository to your local machine.
2. Install the required dependencies.
3. Use the methods provided in `TransferSHAP.py` to calculate transfer feature importance.
4. Use the `cross_statistics.py` script to calculate statistics on the output files generated 
by the TransferSHAP method.

For more detailed instructions, please refer to the code documentation and examples in the repository.

Please note that this package is still under development and may be subject to updates and 
improvements in the future.

For any questions or issues, please contact the authors.
