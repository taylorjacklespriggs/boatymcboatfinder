#!/bin/bash
set -e
if [ -d train ]; then
  exit 0
fi
aws s3 cp s3://taylor-hackathon9/boat_data.zip .
unzip boat_data.zip
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
