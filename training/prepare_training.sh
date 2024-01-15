#!/bin/sh
# prepare_training.sh
# author: Julie Kallini

readonly MISTRAL_PATH=/nlp/scr/kallini/mistral

echo "
-------------------------------------------------------------------------------
Arguments
-------------------------------------------------------------------------------
"
echo "Perturbation type: $1"
echo "Train set: $2"
echo "Random seed: $3"
echo "Paren pretrained model: $4"
NO_POS_ENCODINGS=${5:-''}
echo "No pos encodings: $NO_POS_ENCODINGS"
echo "Mistral path: $MISTRAL_PATH"

if [ -z "$NO_POS_ENCODINGS" ]
then
    NPS=""
    NPSunderscore=""
else
    NPS="-no-positional-encodings"
    NPSunderscore="_no_positional_encodings"
fi

# Generate yaml files for mistral training
echo "
-------------------------------------------------------------------------------
Generating yaml files for mistral training
-------------------------------------------------------------------------------
"
GENERATE_YAML_COMMAND="python3 generate_yaml.py $1 $2 $3 $4 $NO_POS_ENCODINGS"
echo $GENERATE_YAML_COMMAND
eval $GENERATE_YAML_COMMAND

# Copy yaml files to mistral directory
echo "
-------------------------------------------------------------------------------
Copying config yaml files to mistral directory
-------------------------------------------------------------------------------
"
COPY_DATASET_COMMAND="cp conf/babylm_$1_$2_$4$NPSunderscore/seed$3/dataset_$1_$2_seed$3.yaml $MISTRAL_PATH/conf/datasets/dataset_$1_$2_seed$3.yaml"
echo $COPY_DATASET_COMMAND
eval $COPY_DATASET_COMMAND
echo ""

COPY_TRAIN_COMMAND="cp conf/babylm_$1_$2_$4$NPSunderscore/seed$3/train_$1_$2_$4$NPSunderscore"_"seed$3.yaml $MISTRAL_PATH/conf/train_$1_$2_$4$NPSunderscore"_"seed$3.yaml"
echo $COPY_TRAIN_COMMAND
eval $COPY_TRAIN_COMMAND
echo ""

COPY_MODEL_COMMAND="cp conf/babylm_$1_$2_$4$NPSunderscore/gpt2$NPS-small-$1-$4.yaml $MISTRAL_PATH/conf/models/gpt2$NPS-small-$1-$4.yaml"
echo $COPY_MODEL_COMMAND
eval $COPY_MODEL_COMMAND

echo "
-------------------------------------------------------------------------------
Done!
-------------------------------------------------------------------------------
"
