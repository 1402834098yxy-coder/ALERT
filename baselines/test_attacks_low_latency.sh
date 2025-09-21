DATASETS=("nytimes")
TEST_TIMES=5
KWS_UNI_SIZE=3000
KWS_EXTRACTION="sorted"

for dataset in "${DATASETS[@]}"; do
    python test_attacks_low_latency.py \
        --dataset "$dataset" \
        --test_times "$TEST_TIMES" \
        --kws_uni_size "$KWS_UNI_SIZE" \
        --kws_extraction "$KWS_EXTRACTION"

    python generate_test_attacks_low_latency.py \
        --dataset "$dataset" \
        --test_times "$TEST_TIMES" \
        --kws_uni_size "$KWS_UNI_SIZE" \
        --kws_extraction "$KWS_EXTRACTION"
    python generate_result_low_latency.py \
        --dataset "$dataset" \
        --kws_uni_size "$KWS_UNI_SIZE"
done


