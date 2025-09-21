#!/bin/bash
# set project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." &> /dev/null && pwd )"
cd "${PROJECT_ROOT}"

DATASETS=('enron')
FACTOR='1'
GPU_NUM="0"
FEATURE_NUM=3000
EXPERIMENT_SETTING="partial"
COUNTERMEASURES="None"
ADP=false
FORCE_RECOMPUTE=true
 
FPBP_DEL_PERCENTAGES=(0)


RUN_TIME=1
for run in $(seq 1 $RUN_TIME)
do
for DATASET in "${DATASETS[@]}"
do
    for FPBP_DEL_PERCENTAGE in "${FPBP_DEL_PERCENTAGES[@]}"
    do
        echo "FPBP_DEL_PERCENTAGE: $FPBP_DEL_PERCENTAGE"

# done

START_PERCENTAGE=3
END_PERCENTAGE=3
STEP_PERCENTAGE=2   
START_DECIMAL=$(echo "$START_PERCENTAGE / 100" | bc -l)
END_DECIMAL=$(echo "$END_PERCENTAGE / 100" | bc -l)
STEP_DECIMAL=$(echo "$STEP_PERCENTAGE / 100" | bc -l)
echo "Generated candidates: ${START_DECIMAL%.2f}-1.0"

# printf '%s' "${candidates[@]}"

python ${PROJECT_ROOT}/src/process_dataset.py \
    --dataset_name "$DATASET" \
    --experiment_setting "$EXPERIMENT_SETTING" \
    --countermeasures "$COUNTERMEASURES" \
    --keyword_size "$FEATURE_NUM" \
    --adp "$ADP" \
    --force_recompute "$FORCE_RECOMPUTE" \
    --start_percentage "$START_PERCENTAGE" \
    --end_percentage "$END_PERCENTAGE" \
    --step_percentage "$STEP_PERCENTAGE" \
    --fpbp_del_percentage "$FPBP_DEL_PERCENTAGE"


START_PERCENTAGE=5
END_PERCENTAGE=20
STEP_PERCENTAGE=5   
START_DECIMAL=$(echo "$START_PERCENTAGE / 100" | bc -l)
END_DECIMAL=$(echo "$END_PERCENTAGE / 100" | bc -l)
STEP_DECIMAL=$(echo "$STEP_PERCENTAGE / 100" | bc -l)
echo "Generated candidates: ${START_DECIMAL%.2f}-1.0"

# printf '%s' "${candidates[@]}"

python ${PROJECT_ROOT}/src/process_dataset.py \
    --dataset_name "$DATASET" \
    --experiment_setting "$EXPERIMENT_SETTING" \
    --countermeasures "$COUNTERMEASURES" \
    --keyword_size "$FEATURE_NUM" \
    --adp "$ADP" \
    --force_recompute "$FORCE_RECOMPUTE" \
    --start_percentage "$START_PERCENTAGE" \
    --end_percentage "$END_PERCENTAGE" \
    --step_percentage "$STEP_PERCENTAGE" \
    --fpbp_del_percentage "$FPBP_DEL_PERCENTAGE"

candidates=("0.03-1.0" "0.05-1.0" "0.1-1.0" "0.2-1.0")



BASE_INPUT_DIR="${PROJECT_ROOT}/dataset/processed/${DATASET}/Accumulation/"
BASE_OUTPUT_DIR_NORM_train="${PROJECT_ROOT}/log/${DATASET}_norm_add_del_new_train_1_${FPBP_DEL_PERCENTAGE}"
BASE_OUTPUT_DIR_train="${PROJECT_ROOT}/log/${DATASET}_add_del_new_train_1_${FPBP_DEL_PERCENTAGE}"
BASE_OUTPUT_DIR_NORM_test="${PROJECT_ROOT}/log/${DATASET}_norm_add_del_new_test_1_${FPBP_DEL_PERCENTAGE}"
BASE_OUTPUT_DIR_test="${PROJECT_ROOT}/log/${DATASET}_add_del_new_test_1_${FPBP_DEL_PERCENTAGE}"

for candidate in "${candidates[@]}"
do
    OUTPUT_DIR_NORM_train="${BASE_OUTPUT_DIR_NORM_train}_${candidate}_${COUNTERMEASURES}_${FACTOR}_${FPBP_DEL_PERCENTAGE}.pkl"  
    OUTPUT_DIR_train="${BASE_OUTPUT_DIR_train}_${candidate}_${COUNTERMEASURES}_${FACTOR}_${FPBP_DEL_PERCENTAGE}.pkl"
    OUTPUT_DIR_NORM_test="${BASE_OUTPUT_DIR_NORM_test}_${candidate}_${COUNTERMEASURES}_${FACTOR}_${FPBP_DEL_PERCENTAGE}.pkl"
    OUTPUT_DIR_test="${BASE_OUTPUT_DIR_test}_${candidate}_${COUNTERMEASURES}_${FACTOR}_${FPBP_DEL_PERCENTAGE}.pkl"

    #check if all output files exist
    if [[ -f "$OUTPUT_DIR_NORM_train" && -f "$OUTPUT_DIR_train" && \
          -f "$OUTPUT_DIR_NORM_test" && -f "$OUTPUT_DIR_test" ]]; then
        echo "All output files exist, skip preprocess: $candidate"
        continue
    fi

    #preprocess
    INPUT_DIR="${BASE_INPUT_DIR}${candidate}/train/${COUNTERMEASURES}"
    # OUTPUT_DIR_NORM_train="${BASE_OUTPUT_DIR_NORM_train}_${candidate}_${COUNTERMEASURES}_${FACTOR}.pkl"  
    # OUTPUT_DIR_train="${BASE_OUTPUT_DIR_train}_${candidate}_${COUNTERMEASURES}_${FACTOR}.pkl"

    python ${PROJECT_ROOT}/src/preprocess_clean.py \
    --input_dir "$INPUT_DIR" \
    --output_dir_norm "$OUTPUT_DIR_NORM_train" \
    --output_dir "$OUTPUT_DIR_train" \
    --dataset "$DATASET" \
    --feature_num "$FEATURE_NUM"

    INPUT_DIR_test="${BASE_INPUT_DIR}${candidate}/test/${COUNTERMEASURES}"
    # OUTPUT_DIR_NORM_test="${BASE_OUTPUT_DIR_NORM_test}_${candidate}_${COUNTERMEASURES}_${FACTOR}.pkl"
    # OUTPUT_DIR_test="${BASE_OUTPUT_DIR_test}_${candidate}_${COUNTERMEASURES}_${FACTOR}.pkl"

    python ${PROJECT_ROOT}/src/preprocess_clean.py \
    --input_dir "$INPUT_DIR_test" \
    --output_dir_norm "$OUTPUT_DIR_NORM_test" \
    --output_dir "$OUTPUT_DIR_test" \
    --dataset "$DATASET" \
    --feature_num "$FEATURE_NUM"
done

# BASE_INPUT_DIR='/home/longxiang/SSE/log_new/'
LOG_BASE_DIR="${PROJECT_ROOT}/result/${DATASET}/${EXPERIMENT_SETTING}/"


# create log dir
mkdir -p $LOG_BASE_DIR


# train process

for candidate in "${candidates[@]}"
do
    TRAIN_INPUT="${BASE_OUTPUT_DIR_train}_${candidate}_${COUNTERMEASURES}_${FACTOR}_${FPBP_DEL_PERCENTAGE}.pkl"
    TEST_INPUT="${BASE_OUTPUT_DIR_test}_${candidate}_${COUNTERMEASURES}_${FACTOR}_${FPBP_DEL_PERCENTAGE}.pkl"
    NORM_TRAIN_INPUT="${BASE_OUTPUT_DIR_NORM_train}_${candidate}_${COUNTERMEASURES}_${FACTOR}_${FPBP_DEL_PERCENTAGE}.pkl"
    NORM_TEST_INPUT="${BASE_OUTPUT_DIR_NORM_test}_${candidate}_${COUNTERMEASURES}_${FACTOR}_${FPBP_DEL_PERCENTAGE}.pkl"
    LOG_TRAIN_FILE="${LOG_BASE_DIR}Cat_${EXPERIMENT_SETTING}_${candidate}_${COUNTERMEASURES}_${FACTOR}_${FPBP_DEL_PERCENTAGE}_${run}_train.log"
    LOG_TEST_FILE="${LOG_BASE_DIR}Cat_${EXPERIMENT_SETTING}_${candidate}_${COUNTERMEASURES}_${FACTOR}_${FPBP_DEL_PERCENTAGE}_${run}_test.log"
    MODEL_MAIN_DIR="${PROJECT_ROOT}/model/${DATASET}/${EXPERIMENT_SETTING}/main/${candidate}/${COUNTERMEASURES}/${FACTOR}/${FPBP_DEL_PERCENTAGE}_${run}"
    MODEL_PCA_DIR="${PROJECT_ROOT}/model/${DATASET}/${EXPERIMENT_SETTING}/pca/${candidate}/${COUNTERMEASURES}/${FACTOR}/${FPBP_DEL_PERCENTAGE}_${run}"
    HEATMAP_OUTPUT_BASE_DIR="${PROJECT_ROOT}/final_result/heatmap/${DATASET}/${EXPERIMENT_SETTING}/${candidate}"
    HEATMAP_DATA_DIR="${HEATMAP_OUTPUT_BASE_DIR}/${candidate}_heatmap.json"
    HEATMAP_OUTPUT_DIR="${HEATMAP_OUTPUT_BASE_DIR}/${candidate}_heatmap.pdf"
    mkdir -p $MODEL_MAIN_DIR
    mkdir -p $MODEL_PCA_DIR
    mkdir -p $HEATMAP_OUTPUT_BASE_DIR
    mkdir -p $(dirname "$LOG_TRAIN_FILE")
    mkdir -p $(dirname "$LOG_TEST_FILE")

    python -u ${PROJECT_ROOT}/src/catboost_classification_dynamic_train_heatmap.py \
        --input_dir1 "$TRAIN_INPUT" \
        --input_dir2 "$TEST_INPUT" \
        --norm_input_dir1 "$NORM_TRAIN_INPUT" \
        --norm_input_dir2 "$NORM_TEST_INPUT" \
        --time_sel 30 \
        --delta 0.1 \
        --eta 2 \
        --if_pca 1 \
        --if_dcluster 1 \
        --if_savemodel 1 \
        --if_gpu 0 \
        --gpu_num "$GPU_NUM" \
        --model_main_dir "$MODEL_MAIN_DIR" \
        --model_pca_dir "$MODEL_PCA_DIR" \
        --dataset "$DATASET" \
        --heatmap_output_dir "$HEATMAP_DATA_DIR" \
        | tee "$LOG_TRAIN_FILE"


    # generate heatmap
    python -u ${PROJECT_ROOT}/src/generate_heatmap.py \
        --input_data_dir "$HEATMAP_DATA_DIR" \
        --output_heatmap_dir "$HEATMAP_OUTPUT_DIR"
  


done
done
done
done
