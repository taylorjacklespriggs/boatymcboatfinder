#!/bin/bash
set -e
if ! [ -f boat_data.zip ]; then
  aws s3 cp s3://taylor-hackathon/boat_data.zip .
  unzip boat_data.zip
fi
mkdir -p train test
(
  cd train
  unzip ../train.zip
)
(
  cd test
  unzip ../test.zip
)
