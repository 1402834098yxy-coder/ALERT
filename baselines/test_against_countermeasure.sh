COUNTERMEASURES=("padding_seal" "padding_linear_2" "padding_cluster")
DATASETS=("enron" "nytimes")
TEST_TIMES=5
KWS_UNI_SIZE=3000
KWS_EXTRACTION="sorted"

for COUNTERMEASURE in "${COUNTERMEASURES[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        python test_against_countermeasure.py \
            --countermeasure "$COUNTERMEASURE" \
            --test_times "$TEST_TIMES" \
            --kws_uni_size "$KWS_UNI_SIZE" \
            --datasets "$DATASET" \
            --kws_extraction "$KWS_EXTRACTION"

        python generate_test_against_countermeasures.py \
            --countermeasure "$COUNTERMEASURE" \
            --test_times "$TEST_TIMES" \
            --kws_uni_size "$KWS_UNI_SIZE" \
            --datasets "$DATASET" \
            --kws_extraction "$KWS_EXTRACTION"
    done
done
