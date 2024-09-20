#!/bin/bash
K_FOLD=5
TEST_TIME=20

KNN_START=5
KNN_END=5

ETA=10
PI=0.95

declare -a file_array=(
    "wine"
    "hayes-roth"
    "contraceptive" 
    "penbased"
    "new-thyroid"
    "dermatology"
    "balance"
    "glass"
    "yeast"
    "ecoli"
    "pageblocks"
    "shuttle"
)

cd build
>"../experiment.txt"
echo "$(date +"%Y-%m-%d %H:%M:%S")" >> "../experiment.txt"
echo -e "K_FOLD=$K_FOLD\nTEST_TIME=$TEST_TIME\nKNN=[$KNN_START, $KNN_END]\nETA=$ETA\nPI=$PI" >> "../experiment.txt"
for i in "${file_array[@]}"
do
    for KNN in $(seq $KNN_START $KNN_END)
    do
        CMAKE_OPTIONS="
            -DKNN=${KNN}
            -DK_FOLD=${K_FOLD}
            -DTEST_TIME=${TEST_TIME}
            -DETA=${ETA}
            -DPI=${PI}
        "
        cmake $CMAKE_OPTIONS ..
        make
        echo "========== $i(K=$KNN) ===========" >> "../experiment.txt"
        nohup ./main "$i" >> "../experiment.txt" 2> /dev/null &
        wait $!
    done
done
echo "Finish" >> "../experiment.txt"