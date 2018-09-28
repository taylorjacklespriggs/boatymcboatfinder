#!/bin/bash
set -e
if [ -f data/train_ship_segmentations.csv ]; then
  exit 0
fi
mkdir -p data
(
  cd data
  aws s3 cp s3://taylor-hackathon9/boat_data.zip .
  unzip boat_data.zip
  rm boat_data.zip
  mkdir -p train test
  (
    cd train
    unzip ../train.zip
  )
  rm train.zip
  (
    cd test
    unzip ../test.zip
  )
  rm test.zip
)
