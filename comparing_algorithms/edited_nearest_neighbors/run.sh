#!/bin/bash
declare -a file_array=(
    # "balance"
    # "cleveland"
    # "contraceptive"
    # "dermatology"
    # "glass"
    # "hayes-roth"
    # "movement_libras"
    # "new-thyroid"
    # "optdigits"
    # "pageblocks"
    # "penbased"
    # "satimage"
    "segment"
    # "shuttle"
    # "tae"
    # "texture"
    # "thyroid"
    # "vehicle"
    # "vowel"
    # "wine"
    # "winequality-red"
    # "winequality-white"
    # "yeast"
)

K_FOLD=5
TEST_TIME=22
                                                
MODEL_TYPE="decision_tree"
MIN_SAMPLES_SPLIT=10
MAX_PURITY=0.95

for KNN in 1
do
    CMAKE_OPTIONS="
        -DKNN=${KNN}
        -DK_FOLD=${K_FOLD}
        -DTEST_TIME=${TEST_TIME}
        -DMODEL_TYPE="${MODEL_TYPE}"
        -DMIN_SAMPLES_SPLIT=${MIN_SAMPLES_SPLIT}
        -DMAX_PURITY=${MAX_PURITY}
    "
    cd build
    cmake $CMAKE_OPTIONS ..
    make

    # filename="../experiments/experiment_${KNN}.txt"
    # >"$filename"

    # echo "Start: $(date +"%Y-%m-%d %H:%M:%S")" >> "$filename"
    # echo -e "KNN=$KNN\nK_FOLD=$K_FOLD\nTEST_TIME=$TEST_TIME\nMODEL_TYPE=$MODEL_TYPE\nMIN_SAMPLES_SPLIT=$MIN_SAMPLES_SPLIT\nMAX_PURITY=$MAX_PURITY" >> "$filename"

    for file in "${file_array[@]}"
    do
        # echo "========== $file ===========" >> "$filename"
        # nohup ./main "$file" >> "$filename" 2> /dev/null &
        echo "========== $file ==========="
        ./main "$file"
        wait $!
    done

    # echo "Finish: $(date +"%Y-%m-%d %H:%M:%S")" >> "$filename"
    cd ..
done
