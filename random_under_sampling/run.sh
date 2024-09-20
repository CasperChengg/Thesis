#!/bin/bash
K_FOLD=5
TEST_TIME=20

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

CMAKE_OPTIONS="
    -DK_FOLD=${K_FOLD}
    -DTEST_TIME=${TEST_TIME}
    -DETA=${ETA}
    -DPI=${PI}
"
cd build
cmake $CMAKE_OPTIONS ..
make

>"../experiment.txt"
echo "$(date +"%Y-%m-%d %H:%M:%S")" >> "../experiment.txt"
echo -e "K_FOLD=$K_FOLD\nTEST_TIME=$TEST_TIME\nETA=$ETA\nPI=$PI" >> "../experiment.txt"
for i in "${file_array[@]}"
do
    echo "========== $i ===========" >> "../experiment.txt"
    nohup ./main "$i" >> "../experiment.txt" 2> /dev/null &
    wait $!
done
echo -e "========== Finish ==========" >> "../experiment.txt"