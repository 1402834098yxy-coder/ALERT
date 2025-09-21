DATASETS=("enron" "nytimes")
TEST_TIMES=3
KWS_UNI_SIZE=3000
KWS_EXTRACTION="sorted"

for dataset in "${DATASETS[@]}"; do
    python test_attacks_default.py \
        --dataset "$dataset" \
        --test_times "$TEST_TIMES" \
        --kws_uni_size "$KWS_UNI_SIZE" \
        --kws_extraction "$KWS_EXTRACTION"

    python generate_test_attacks_default.py \
        --dataset "$dataset" \
        --test_times "$TEST_TIMES" \
        --kws_uni_size "$KWS_UNI_SIZE" \
        --kws_extraction "$KWS_EXTRACTION"
done

# python generate_test_attacks_default.py \
#     --dataset "enron" \
#     --desired_time 5 \
#     --test_times 5 \
#     --kws_uni_size 3000 \
#     --kws_extraction "sorted"