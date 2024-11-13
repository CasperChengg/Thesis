#!/bin/bash
declare -a file_array=(
    "wine"
    # "hayes-roth"
    # "contraceptive" 
    # "penbased"
    # "new-thyroid"
    # "dermatology"
    # "balance"
    # "glass"
    # "yeast"
    # "ecoli"
    # "pageblocks"
    # "shuttle"
)

      
K_FOLD=5
TEST_TIME=20
                                                
MODEL_TYPE="decision_tree"
MIN_SAMPLES_SPLIT=10
MAX_PURITY=0.95

CMAKE_OPTIONS="
    -DK_FOLD=${K_FOLD}
    -DTEST_TIME=${TEST_TIME}
    -DMODEL_TYPE="${MODEL_TYPE}"
    -DMIN_SAMPLES_SPLIT=${MIN_SAMPLES_SPLIT}
    -DMAX_PURITY=${MAX_PURITY}
"
cd build
cmake $CMAKE_OPTIONS ..
make

filename="../experiments/experiment_$(date +"%Y-%m-%d_%H-%M").txt"
>"$filename"

echo "Start: $(date +"%Y-%m-%d %H:%M:%S")" >> "$filename"
echo -e "K_FOLD=$K_FOLD\nTEST_TIME=$TEST_TIME\nMODEL_TYPE=$MODEL_TYPE\nMIN_SAMPLES_SPLIT=$MIN_SAMPLES_SPLIT\n" >> "$filename"

for file in "${file_array[@]}"
do
    echo "========== $file ===========" >> "$filename"
    nohup ./main "$file" >> "$filename" 2> /dev/null &
    wait $!
done
echo "Finish: $(date +"%Y-%m-%d %H:%M:%S")" >> "$filename"