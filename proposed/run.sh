#!/bin/bash
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

KNN=5
K_FOLD=5
TEST_TIME=20

ETA=10
PI=0.95

USE_RNN=ON
USE_KNN=OFF
CONSIDER_MAJ=OFF
CONSIDER_MIN=ON
KNN_DISTANCE=OFF
DIVIDE_BY_DISTANCE=ON
MULTIPLE_BY_DISTANCE=OFF

CMAKE_OPTIONS="S
    -DKNN=${KNN}
    -DK_FOLD=${K_FOLD}
    -DTEST_TIME=${TEST_TIME}
    -DETA=${ETA}
    -DPI=${PI}
    -DUSE_KNN=${USE_KNN}
    -DCONSIDER_MAJ=${CONSIDER_MAJ}
    -DCONSIDER_MIN=${CONSIDER_MIN}
    -DKNN_DISTANCE=${KNN_DISTANCE}
    -DDIVIDE_BY_DISTANCE=${DIVIDE_BY_DISTANCE}
    -DMULTIPLE_BY_DISTANCE=${MULTIPLE_BY_DISTANCE}
    -DUSE_RNN=${USE_RNN}
"
cd build
cmake $CMAKE_OPTIONS ..
make

>"../experiment.txt"
echo "Start: $(date +"%Y-%m-%d %H:%M:%S")" >> "../experiment.txt"
echo -e "K_FOLD=$K_FOLD\nTEST_TIME=$TEST_TIME\nKNN=$KNN\nETA=$ETA\nPI=$PI\nUSE_KNN=$USE_KNN\nUSE_RNN=$USE_RNN\nCONSIDER_MAJ=$CONSIDER_MAJ" >> "../experiment.txt"
echo -e "CONSIDER_MIN=$CONSIDER_MIN\nKNN_DISTANCE=$KNN_DISTANCE\nDIVIDE_BY_DISTANCE=$DIVIDE_BY_DISTANCE\nMULTIPLE_BY_DISTANCE=$MULTIPLE_BY_DISTANCE" >> "../experiment.txt"
for i in "${file_array[@]}"
do
    echo "========== $i ===========" >> "../experiment.txt"
    nohup ./main "$i" >> "../experiment.txt" 2> /dev/null &
    wait $!
done
echo "Finish: $(date +"%Y-%m-%d %H:%M:%S")" >> "../experiment.txt"