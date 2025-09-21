SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." &> /dev/null && pwd )"
cd "${PROJECT_ROOT}"

TASK='test_attacks_default'
DATASETS=('enron')
FACTOR='1'
GPU_NUM="0"
FEATURE_NUM=3000
EXPERIMENT_SETTING="sample"
COUNTERMEASURES="None"
ADP=false
FORCE_RECOMPUTE=true
START_PERCENTAGE=50
END_PERCENTAGE=50 
STEP_PERCENTAGE=20    
KWS_EXTRACTION="sorted"
FPBP_DEL_PERCENTAGES=(0)

# transform percentage to decimal
START_DECIMAL=$(echo "$START_PERCENTAGE / 100" | bc -l)
END_DECIMAL=$(echo "$END_PERCENTAGE / 100" | bc -l)
STEP_DECIMAL=$(echo "$STEP_PERCENTAGE / 100" | bc -l)

RUN_TIME=30
for run in $(seq 1 $RUN_TIME)
do
for DATASET in "${DATASETS[@]}"
do
if [ "$DATASET" = "wiki_3000" ] || [ "$DATASET" = "wiki_5000" ] || [ "$DATASET" = "wiki_7000" ] || [ "$DATASET" = "nytimes" ]; then
TIME_SEL=30
TIME_SEL_TEST=30
else
TIME_SEL=31
TIME_SEL_TEST=31
fi
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


echo "Generated candidates:"
printf '%s' "${candidates[@]}"

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

LOG_BASE_DIR="${PROJECT_ROOT}/result/${DATASET}/${EXPERIMENT_SETTING}/"


# create log dir
mkdir -p $LOG_BASE_DIR


# train process

for candidate in "${candidates[@]}"
do
    INPUT_DIR_test="${BASE_INPUT_DIR}${candidate}/test/${COUNTERMEASURES}"
    TRAIN_INPUT="${BASE_OUTPUT_DIR_train}_${candidate}_${COUNTERMEASURES}_${FACTOR}_${FPBP_DEL_PERCENTAGE}.pkl"
    TEST_INPUT="${BASE_OUTPUT_DIR_test}_${candidate}_${COUNTERMEASURES}_${FACTOR}_${FPBP_DEL_PERCENTAGE}.pkl"
    NORM_TRAIN_INPUT="${BASE_OUTPUT_DIR_NORM_train}_${candidate}_${COUNTERMEASURES}_${FACTOR}_${FPBP_DEL_PERCENTAGE}.pkl"
    NORM_TEST_INPUT="${BASE_OUTPUT_DIR_NORM_test}_${candidate}_${COUNTERMEASURES}_${FACTOR}_${FPBP_DEL_PERCENTAGE}.pkl"
    LOG_TRAIN_FILE="${LOG_BASE_DIR}ALERT_${TASK}_${DATASET}_${EXPERIMENT_SETTING}_${candidate}_${COUNTERMEASURES}_${FACTOR}_${FPBP_DEL_PERCENTAGE}_tilde_beta_0.4_${run}_train.log"
    LOG_TEST_FILE="${LOG_BASE_DIR}ALERT_${TASK}_${DATASET}_${EXPERIMENT_SETTING}_${candidate}_${COUNTERMEASURES}_${FACTOR}_${FPBP_DEL_PERCENTAGE}_tilde_beta_0.4_${run}_test.log"
    MODEL_MAIN_DIR="${PROJECT_ROOT}/model/${DATASET}/${EXPERIMENT_SETTING}/main/${candidate}/${COUNTERMEASURES}/${FACTOR}/${FPBP_DEL_PERCENTAGE}_${run}"
    MODEL_PCA_DIR="${PROJECT_ROOT}/model/${DATASET}/${EXPERIMENT_SETTING}/pca/${candidate}/${COUNTERMEASURES}/${FACTOR}/${FPBP_DEL_PERCENTAGE}_${run}"
    mkdir -p $MODEL_MAIN_DIR
    mkdir -p $MODEL_PCA_DIR
    mkdir -p $(dirname "$LOG_FILE")

    python -u ${PROJECT_ROOT}/src/catboost_classification_dynamic_train.py \
        --input_dir1 "$TRAIN_INPUT" \
        --input_dir2 "$TEST_INPUT" \
        --norm_input_dir1 "$NORM_TRAIN_INPUT" \
        --norm_input_dir2 "$NORM_TEST_INPUT" \
        --time_sel "$TIME_SEL" \
        --beta 1 \
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
        | tee "$LOG_TRAIN_FILE"

     python -u ${PROJECT_ROOT}/src/catboost_classification_dynamic_test.py \
        --input_dir1 "$TRAIN_INPUT" \
        --input_dir2 "$TEST_INPUT" \
        --norm_input_dir1 "$NORM_TRAIN_INPUT" \
        --norm_input_dir2 "$NORM_TEST_INPUT" \
        --time_sel "$TIME_SEL" \
        --beta 1 \
        --beta_test 0.4 \
        --delta 0.4 \
        --eta 2 \
        --if_pca 1 \
        --if_dcluster 1 \
        --gpu_num "$GPU_NUM" \
        --model_main_dir "$MODEL_MAIN_DIR" \
        --model_pca_dir "$MODEL_PCA_DIR" \
        --dataset "$DATASET" \
        --preprocess_input_dir "$INPUT_DIR_test" \
        |    tee "$LOG_TEST_FILE"
    

    done

done
done
done


OUTPUT_folder="${PROJECT_ROOT}/final_result/${TASK}/"
mkdir -p $OUTPUT_folder
for DATASET in "${DATASETS[@]}"
do
for FPBP_DEL_PERCENTAGE in "${FPBP_DEL_PERCENTAGES[@]}"
do
  LOG_BASE_FILE="${PROJECT_ROOT}/result/${DATASET}/${EXPERIMENT_SETTING}/ALERT_${TASK}_${DATASET}_${EXPERIMENT_SETTING}_${candidate}_${COUNTERMEASURES}_${FACTOR}_${FPBP_DEL_PERCENTAGE}_tilde_beta_0.4_"


        
PART='test'
OUTPUT_DIR="${OUTPUT_folder}/ALERT_${DATASET}_${EXPERIMENT_SETTING}_${candidate}_${PART}_"
python ${PROJECT_ROOT}/src/generate_result.py \
    --log_base_dir $LOG_BASE_FILE \
    --runtime $RUN_TIME \
    --part $PART \
    --output_dir $OUTPUT_DIR \
    --keyword_size $FEATURE_NUM
done
done



