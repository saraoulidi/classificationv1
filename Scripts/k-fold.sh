#!/bin/bash

#This script should be executed from the root dir of the project
#the main idea of this code is to do Kfold cross validation with bash scripting

clear

datasetsBaseDir=$1  # without the trailing slash plz
NbrOfFolds=$2
NbrOfRandoms=$3
TrainingSlice=$4
ValidationSlice=$(( $3-$4 ))
NbrOfEpochs=$5
Optimizer=$6
LearningRate=$7
Dropout1=$8
Dropout2=$9

echo "Using dataset from $datasetsBaseDir"
echo "I will do $NbrOfFolds folds"
echo "Random sample size $NbrOfRandoms"
echo "Number of training images $TrainingSlice"
echo "I will validate with $ValidationSlice images"
echo "Epochs $NbrOfEpochs"
echo "I will use $Optimizer as and optimizer"
echo "My learning rate will be $LearningRate"
echo "Dropout1 treshhold $Dropout1"
echo "Dropout2 treshhold $Dropout2"
sleep 5;

for ((kf=1; kf<=$NbrOfFolds; kf++))

do

  echo "Doing fold Number $kf"
  modelName="kf-model-$kf-$(uuidgen)"
  reportDir="./kfold-reports/$modelName"
  mkdir -p $reportDir

  echo "Prepare data!!"

  for brandPath in $datasetsBaseDir/*; do
    # select brandname
    brandName=$(basename $brandPath)

    if [ -d $brandPath ]; then # ignore any file in the directory

      echo "Fold $brandName!!"

      # select a random set
      randomSet=$(ls -1 $brandPath/* | sort -R)
      randomSetArray=(${randomSet//$' '/ })

      IFS=$'\n'; # set separator

      # select training images from the top
      trainingImages=$(echo "${randomSetArray[*]}" | head -$NbrOfRandoms | head -$TrainingSlice)

      # select validtion images from the bottom
      validationImages=$(echo "${randomSetArray[*]}" | head -$NbrOfRandoms | tail -$ValidationSlice)

      # create the current brand train and validation dir.
      mkdir -p ./kfold-dataset/{train,validation}/$brandName

      # copy training images to the train dir
      cp $trainingImages ./kfold-dataset/train/$brandName

      # copy validation images to the validation dir
      cp $validationImages ./kfold-dataset/validation/$brandName

      # save images that lead to the current accuracy
      echo $trainingImages > $reportDir/$brandName-images.txt

    fi

  done

  # print out the used traning command
  echo "Run training!!"
  echo "python3 ./t.py $modelName -p ./kfold-dataset/ -ep $NbrOfEpochs -b 25 -o $Optimizer -l $LearningRate -d1 $Dropout1 -d2 $Dropout2"

  # select the validation accuracy
  kfacc=$(python3 ./t.py $modelName -p ./kfold-dataset/ -ep $NbrOfEpochs -b 25 -o $Optimizer -l $LearningRate -d1 $Dropout1 -d2 $Dropout2 | tail -1)

  echo "Save report!!"
  mv -v $reportDir "./kfold-reports/$kfacc-$modelName"

  # cleanup
  echo "Do cleanup!!"
  rm -rf ./kfold-dataset/train/* ./kfold-dataset/validation/*
  rm -f ./Models/$modelName.h5

done

# open Tensorboard
tensorboard --logdir="logs/"

exit 0
