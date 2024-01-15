#!/bin/sh
# perturb.sh
# author: Julie Kallini

echo "
-------------------------------------------------------------------------------
Arguments
-------------------------------------------------------------------------------
"
echo "Perturbation type: $1"
echo "Train set: $2"


# Create perturbed dataset for all splits
echo "
-------------------------------------------------------------------------------
Creating perturbed dataset for all splits
-------------------------------------------------------------------------------
"

cd ../data

echo "python3 perturb.py $1 $2"
python3 perturb.py $1 $2
echo "
python3 perturb.py $1 dev"
python3 perturb.py $1 dev
echo "
python3 perturb.py $1 test"
python3 perturb.py $1 test
echo "
python3 perturb.py $1 unittest"
python3 perturb.py $1 unittest

cd ..