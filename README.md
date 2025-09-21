# ALERT
This repository contains the code to evaluate ALERT in Dynamic Searchable Symmetric Encryption (DSSE) schemes.

This code can be used to reproduce the results in this paper:
**"ALERT: Machine Learning-Enhanced Risk Estimation for Databases Supporting Encrypted Queries"**.

**DISCLAIMER**: the code should work, but it's not very efficient, there is still a lot of room for improvement.

## Folders
```
.
├── baselines/
├── dataset/
├── final_result/
├── log/
├── model/
├── result/
├── scripts/
└── src/
```

The `baselines` folder contains code for evaluating results from three baseline LAAs. The `dataset` folder stores preprocessed documents for searchable encryption. The `final_result`, `log`, `model`, and `result` folders contain essential results and data (e.g., trained models, extracted co-occurrence matrices) in the risk assessment process. The `scripts` folder provides scripts for launching risk assessments across different SSE schemes and adversary assumptions. The `src` folder contains ALERT's core source code, including data preprocessing utilities, classifier training modules, and query risk assessment implementations.

## Quickstart
To ensure full compatibility with ALERT, we recommend creating a new conda environment with Python 3.9.16:
```
conda create -n alert python=3.9.16
conda activate alert
```
Next, install the required dependencies:

```
pip install -r requirements.txt
```

Then, you can reproduce our experiments with our scripts.
```
cd scripts
```

**PS**: The implementation of ALERT is fundamentally GPU-based, so hardware specifications (GPU/CPU) may show variations in risk assessment latency measurements. These variations are expected and do not significantly impact the overall validity of this paper's findings.

## About the Datasets

The folder `dataset` contains the datasets needed for reproducing our results.

For Enron, the `enron_db.pkl` contains a document list and a keyword dict. The document list contains lists of keywords for each file. The keyword dict maps each keyword to its total counts in files and query trends.  Although these query trends are not utilized in ALERT, they are retained to maintain compatibility with baseline LAAs. The same structure is used for NYTimes `nytimes_db.pkl` and Wikipedia `kws_num_new_0.pkl, kws_num_5000.pkl,kws_num_7000.pkl`. 
All datasets are processed into a uniform format using `src/preprocess_dataset.py`.



## Main Recovery Results
The main results are illustrated in Figure 4 in the paper, which demonstrates ALERT's effectiveness across different datasets and adversary settings.


Before running the script, you are required to determine the classifier training mode. We provide a field "if_gpu" in each script to determine if use the GPU for training.
If using GPU mode, you need to further check the GPU configurations. The `GPU_NUM` in `main_recovery_partial.sh` and `main_recovery_sample.sh` should be set to a proper number to direct the application to the desired GPU device. This GPU configuration check is required for all following experiments.

```
bash main_recovery.sh
```

Here, the final results are stored in the `final_result/main_recovery` folder. Each file contains a "cumulative_accuracy" field that records recovery rates for different keyword numbers. The filename indicates the experimental settings: dataset names (Enron/NYTimes/Wiki), adversary type (partial/sample), and adversary assumptions (data partial known rate $\gamma$, and data sampling rate $\alpha$).

## An example of Recovery Map of ALERT

This experiment demonstrates a visual representation of ALERT (Figure 3 in paper). It's important to note that risk assessment results for concrete queries may exhibit randomness across different training sessions due to varying initialization parameters. Therefore, we provide specific training results (probability matrices) for generating the heat map.

To generate the heatmap yourself, follow these steps:

```
bash test_heatmap.sh
```
Then, you can check the heatmap drawn by different data partial known rates $\gamma$ in `final_result/heatmap`. We need to clarify that this example aims to illustrate our findings about risk assessment. Due to the randomness in training, the results may differ from those presented in the paper. Therefore, we provide the original data and heatmaps for reference in `final_result/heatmap/ref`.

## Keyword Clustering Results with Fixed Parameters

This experiment validates the effectiveness of keyword clustering based on volume information (as shown in Figure 5). To reproduce the experiment, you should first rerun the script of `main_recovery` to get the co-occurrence log, which is required for this experiment. If you want to independently reproduce the results of this experiment, we also provide a script to generate these co-occurrence logs.
```
bash test_dcm_fixed_param.sh
```

Then, you can reproduce results based on the provided ipynb file `scripts/generate_result_dcm_fixed_param.ipynb`. The notebook contains three sections, each corresponding to a different subfigure in Figure 5, demonstrating the effects of different keyword numbers ($\vartheta$), various sampling rates ($\alpha$), and different clustering thresholds ($\theta$). Each section in the file includes the clustering accuracy needed for the two datasets.



## Performance with/without Dynamic Keyword Clustering Mechanism
The experiment corresponds to Table 1 in the paper. Here, we compare the recovery rates and program runtime with and without the mechanism. Here is the script how to run the experiment
```
bash test_dcm.sh
```
The results are stored in `final_result/test_dcm`.

## Performance across Forward/Backward Privacy-DSSE

The performance across FP/BP-DSSE is shown in Figure 6 in this paper. Here, we vary different data deletion rates $\lambda$ and show its impact on risk assessment results.

```
bash test_fpbp.sh
```

Like above, the results of the experiment are stored in `final_result/test_fpbp` folder, demonstrating the recovery rate under different data deletion rates $\lambda$.

## Comparisons with Prior Alternatives

To validate the effectiveness of ALERT, this paper compares it with three baseline LAAs: Jigsaw attack, RSA attack, and IHOP attack. We reproduce these baseline attacks using the Jigsaw implementation (https://github.com/JigsawAttack/JigsawAttack). Each method is executed 30 times to generate boxplots.

**PS:** Although ALERT performs a quick risk assessment, the training process typically takes about 1-2 hours. For quick validation of the main claims, you can modify the `RUN_TIME=5` (which is the least repeated time needed to draw a box plot) in the scripts in this Section (Comparisons with Prior Alternatives). Note that this simplification may lead to increased variance in the results.

### Comparisons Under Low-latency Scenario
In this section, we adjust the convergence speed parameters for each method: `RefSpeed` for Jigsaw and RSA, `n_iters` for IHOP, and $\tilde{\beta}$ for ALERT. These adjustments control the program runtime and risk assessment time. Here is the script to reproduce the experiment, which corresponds to Figure 7 in the paper:

```
bash test_attacks_low_latency.sh
```
The results are reserved in `final_result/test_attacks_low_latency` folder, where each filename indicates the corresponding dataset, attack method, and desired risk assessment latency. Each file contains data for box plot generation, including minimum value, first quartile (Q1, 25th percentile), median, third quartile (Q3, 75th percentile), maximum value, and the average runtime in seconds.

### Comparisons without Time Constraints
This section aims to compare the effectiveness of different risk analysis methods without runtime constraints, corresponding to Figure 8 in the paper. Here is the script to reproduce the experiment:
```
bash test_attacks_default.sh
```
Correspondingly, the results are reserved in `final_result/test_attacks_default`. Note that we set $\tilde{\beta} = 1$ to maximize the risk assessment recovery accuracy when operating without time constraints. For additional validation under default settings, we provide another script with $\tilde{\beta} = 0.4$:
```
bash test_attacks_default_extend.sh
```

### Comparisons in Larger Keyword Universe Sizes under Similar Time Constraints
We compare ALERT and Jigsaw in the Wikipedia dataset to evaluate their effectiveness when facing larger keyword universe sizes. We set ALERT with $\tilde{\beta}=0.2, \tilde{\delta}=0.2, \tilde{\eta}=1$. This experiment corresponds to Figure 9 in the paper.
To run the experiment:

```
bash test_large_keyword_wiki.sh
```
The results are stored in `final_result/test_large_keyword_wiki`.

### Comparisons Against Countermeasures

Here, we evaluate ALERT and baseline LAAs against three padding countermeasures, as illustrated in Figure 10 in the paper. Each risk analysis method is constrained to a 10-second runtime limit. To facilitate reproduction, we provide a script that executes the complete evaluation process:

``` 
bash test_against_countermeasures.sh
```

The experimental results are stored in the `final_result/test_against_countermeasures` directory. Each filename contains the corresponding dataset name, risk analysis method, and applied countermeasure. Like the above experiments, each file contains the necessary results for generating box plots.





