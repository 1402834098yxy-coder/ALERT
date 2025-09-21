#!/bin/bash
# set project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." &> /dev/null && pwd )"
cd "${PROJECT_ROOT}"
TASK='test_dcm_fixed_param'
DATASETS=('nytimes' 'enron')
FACTOR='1'
GPU_NUM="0"
FEATURE_NUM=3000
EXPERIMENT_SETTING="sample"
COUNTERMEASURES="None"
ADP=false
FORCE_RECOMPUTE=true
  
FPBP_DEL_PERCENTAGES=(0)
# transform percentage to decimal
START_PERCENTAGE=5
END_PERCENTAGE=50
STEP_PERCENTAGE=5  
START_DECIMAL=$(echo "$START_PERCENTAGE / 100" | bc -l)
END_DECIMAL=$(echo "$END_PERCENTAGE / 100" | bc -l)
STEP_DECIMAL=$(echo "$STEP_PERCENTAGE / 100" | bc -l)

RUN_TIME=1
for run in $(seq 1 $RUN_TIME)
do
for DATASET in "${DATASETS[@]}"
do
    for FPBP_DEL_PERCENTAGE in "${FPBP_DEL_PERCENTAGES[@]}"
    do
        echo "FPBP_DEL_PERCENTAGE: $FPBP_DEL_PERCENTAGE"
    if [ "$EXPERIMENT_SETTING" = "sample" ]; then
    # Original sample logic - generates pairs like 0.1-0.9, 0.2-0.8, etc.
    candidates=()
    current=$START_DECIMAL
        while (( $(echo "$current <= $END_DECIMAL" | bc -l) )); do
            opposite=$(echo "1 - $current" | bc -l)
            
            if (( $(echo "($current * 100) % 10 == 0" | bc) )); then
                formatted_pair=$(printf "%.1f-%.1f" $current $opposite)
            else
                formatted_pair=$(printf "%.2f-%.2f" $current $opposite)
            fi
            
            candidates+=("$formatted_pair")
            current=$(echo "$current + $STEP_DECIMAL" | bc -l)
        done
    else
    # Partial experiment logic - generates pairs like 0.1-1.0, 0.2-1.0, etc.
    candidates=()
    current=$START_DECIMAL
    while (( $(echo "$current <= $END_DECIMAL" | bc -l) )); do
        if (( $(echo "($current * 100) % 10 == 0" | bc) )); then
            formatted_pair=$(printf "%.1f-1.0" $current)
        else
            formatted_pair=$(printf "%.2f-1.0" $current)
        fi
        
        candidates+=("$formatted_pair")
            current=$(echo "$current + $STEP_DECIMAL" | bc -l)
    done
    fi


# echo "Generated candidates:"

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


BASE_INPUT_DIR="${PROJECT_ROOT}/dataset/processed/${DATASET}/Accumulation/"
BASE_OUTPUT_DIR_NORM_train="${PROJECT_ROOT}/log/${DATASET}_norm_add_del_new_train_1_${FPBP_DEL_PERCENTAGE}"
BASE_OUTPUT_DIR_train="${PROJECT_ROOT}/log/${DATASET}_add_del_new_train_1_${FPBP_DEL_PERCENTAGE}"
BASE_OUTPUT_DIR_NORM_test="${PROJECT_ROOT}/log/${DATASET}_norm_add_del_new_test_1_${FPBP_DEL_PERCENTAGE}"
BASE_OUTPUT_DIR_test="${PROJECT_ROOT}/log/${DATASET}_add_del_new_test_1_${FPBP_DEL_PERCENTAGE}"

candidates=("0.05-0.95" "0.1-0.9" "0.2-0.8" "0.3-0.7" "0.5-0.5")


for candidate in "${candidates[@]}"
do
    OUTPUT_DIR_NORM_train="${BASE_OUTPUT_DIR_NORM_train}_${candidate}_${COUNTERMEASURES}_${FACTOR}_${FPBP_DEL_PERCENTAGE}.pkl"  
    OUTPUT_DIR_train="${BASE_OUTPUT_DIR_train}_${candidate}_${COUNTERMEASURES}_${FACTOR}_${FPBP_DEL_PERCENTAGE}.pkl"
    OUTPUT_DIR_NORM_test="${BASE_OUTPUT_DIR_NORM_test}_${candidate}_${COUNTERMEASURES}_${FACTOR}_${FPBP_DEL_PERCENTAGE}.pkl"
    OUTPUT_DIR_test="${BASE_OUTPUT_DIR_test}_${candidate}_${COUNTERMEASURES}_${FACTOR}_${FPBP_DEL_PERCENTAGE}.pkl"


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
done
done
done
