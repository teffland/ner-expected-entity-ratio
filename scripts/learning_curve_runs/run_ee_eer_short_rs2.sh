# Learning curve experiment: EER on documents with annotations of varying coverage, rs2
RUN_JUPYTER=false
RUN_TENSORBOARD=false
LANG_LABEL=eng-c
METHOD_LABEL=eer-short
BASE_CONFIG=experiments/supervised_tagger_short.jsonnet
ASSUME_COMPLETE="false"
LANG_DIR=data/conll2003/eng
DEV_DATA=entity.dev-docs.jsonl
TEST_DATA=entity.test-docs.jsonl
VOCAB_PATH=data/conll2003/roberta-entity.vocab
MODEL_NAME=roberta-base
PAD_TOKEN="<pad>"
OOV_TOKEN="<unk>"

BATCH_SIZE=15
RANDOM_SEED=2
DROPOUT=0.2
LR=2e-5
NUM_EPOCHS=50
PRIOR_TYPE="eer-exact"
PRIOR_WEIGHT=10.0
ENTITY_RATIO=0.15
ENTITY_RATIO_MARGIN=0.05


# # tacl-eer_eng-c_P-100_rs2_eer
# DATASET_LABEL=P-100_rs$RANDOM_SEED
# bash scripts/run_remote_ml_experiment.sh\
#  PUBLIC_IP=35.173.224.40\
#  PRIVATE_IP=172.31.59.108\
#  RUN_JUPYTER=$RUN_JUPYTER\
#  RUN_TENSORBOARD=$RUN_TENSORBOARD\
#  LANG_LABEL=$LANG_LABEL\
#  DATASET_LABEL=$DATASET_LABEL\
#  METHOD_LABEL=$METHOD_LABEL\
#  BASE_CONFIG=$BASE_CONFIG\
#  ASSUME_COMPLETE=$ASSUME_COMPLETE\
#  LANG_DIR=$LANG_DIR\
#  TRAIN_DATA=entity.train-docs_$DATASET_LABEL.jsonl\
#  DEV_DATA=$DEV_DATA\
#  TEST_DATA=$TEST_DATA\
#  VOCAB_PATH=$VOCAB_PATH\
#  MODEL_NAME=$MODEL_NAME\
#  PAD_TOKEN=$PAD_TOKEN\
#  OOV_TOKEN=$OOV_TOKEN\
#  BATCH_SIZE=$BATCH_SIZE\
#  VALIDATION_BATCH_SIZE=$BATCH_SIZE\
#  RANDOM_SEED=$RANDOM_SEED\
#  DROPOUT=$DROPOUT\
#  LR=$LR\
#  NUM_EPOCHS=$NUM_EPOCHS\
#  PRIOR_TYPE=$PRIOR_TYPE\
#  PRIOR_WEIGHT=$PRIOR_WEIGHT\
#  ENTITY_RATIO=$ENTITY_RATIO\
#  ENTITY_RATIO_MARGIN=$ENTITY_RATIO_MARGIN


# # tacl-eer_eng-c_P-500_rs2_eer
# DATASET_LABEL=P-500_rs$RANDOM_SEED
# bash scripts/run_remote_ml_experiment.sh\
#  PUBLIC_IP=44.192.19.229\
#  PRIVATE_IP=172.31.61.42\
#  RUN_JUPYTER=$RUN_JUPYTER\
#  RUN_TENSORBOARD=$RUN_TENSORBOARD\
#  LANG_LABEL=$LANG_LABEL\
#  DATASET_LABEL=$DATASET_LABEL\
#  METHOD_LABEL=$METHOD_LABEL\
#  BASE_CONFIG=$BASE_CONFIG\
#  ASSUME_COMPLETE=$ASSUME_COMPLETE\
#  LANG_DIR=$LANG_DIR\
#  TRAIN_DATA=entity.train-docs_$DATASET_LABEL.jsonl\
#  DEV_DATA=$DEV_DATA\
#  TEST_DATA=$TEST_DATA\
#  VOCAB_PATH=$VOCAB_PATH\
#  MODEL_NAME=$MODEL_NAME\
#  PAD_TOKEN=$PAD_TOKEN\
#  OOV_TOKEN=$OOV_TOKEN\
#  BATCH_SIZE=$BATCH_SIZE\
#  VALIDATION_BATCH_SIZE=$BATCH_SIZE\
#  RANDOM_SEED=$RANDOM_SEED\
#  DROPOUT=$DROPOUT\
#  LR=$LR\
#  NUM_EPOCHS=$NUM_EPOCHS\
#  PRIOR_TYPE=$PRIOR_TYPE\
#  PRIOR_WEIGHT=$PRIOR_WEIGHT\
#  ENTITY_RATIO=$ENTITY_RATIO\
#  ENTITY_RATIO_MARGIN=$ENTITY_RATIO_MARGIN


# # tacl-eer_eng-c_P-1000_rs2_eer
# DATASET_LABEL=P-1000_rs$RANDOM_SEED
# bash scripts/run_remote_ml_experiment.sh\
#  PUBLIC_IP=3.231.226.31\
#  PRIVATE_IP=172.31.48.111\
#  RUN_JUPYTER=$RUN_JUPYTER\
#  RUN_TENSORBOARD=$RUN_TENSORBOARD\
#  LANG_LABEL=$LANG_LABEL\
#  DATASET_LABEL=$DATASET_LABEL\
#  METHOD_LABEL=$METHOD_LABEL\
#  BASE_CONFIG=$BASE_CONFIG\
#  ASSUME_COMPLETE=$ASSUME_COMPLETE\
#  LANG_DIR=$LANG_DIR\
#  TRAIN_DATA=entity.train-docs_$DATASET_LABEL.jsonl\
#  DEV_DATA=$DEV_DATA\
#  TEST_DATA=$TEST_DATA\
#  VOCAB_PATH=$VOCAB_PATH\
#  MODEL_NAME=$MODEL_NAME\
#  PAD_TOKEN=$PAD_TOKEN\
#  OOV_TOKEN=$OOV_TOKEN\
#  BATCH_SIZE=$BATCH_SIZE\
#  VALIDATION_BATCH_SIZE=$BATCH_SIZE\
#  RANDOM_SEED=$RANDOM_SEED\
#  DROPOUT=$DROPOUT\
#  LR=$LR\
#  NUM_EPOCHS=$NUM_EPOCHS\
#  PRIOR_TYPE=$PRIOR_TYPE\
#  PRIOR_WEIGHT=$PRIOR_WEIGHT\
#  ENTITY_RATIO=$ENTITY_RATIO\
#  ENTITY_RATIO_MARGIN=$ENTITY_RATIO_MARGIN


# # tacl-eer_eng-c_P-5000_rs2_eer
# DATASET_LABEL=P-5000_rs$RANDOM_SEED
# bash scripts/run_remote_ml_experiment.sh\
#  PUBLIC_IP=3.238.112.212\
#  PRIVATE_IP=172.31.55.137\
#  RUN_JUPYTER=$RUN_JUPYTER\
#  RUN_TENSORBOARD=$RUN_TENSORBOARD\
#  LANG_LABEL=$LANG_LABEL\
#  DATASET_LABEL=$DATASET_LABEL\
#  METHOD_LABEL=$METHOD_LABEL\
#  BASE_CONFIG=$BASE_CONFIG\
#  ASSUME_COMPLETE=$ASSUME_COMPLETE\
#  LANG_DIR=$LANG_DIR\
#  TRAIN_DATA=entity.train-docs_$DATASET_LABEL.jsonl\
#  DEV_DATA=$DEV_DATA\
#  TEST_DATA=$TEST_DATA\
#  VOCAB_PATH=$VOCAB_PATH\
#  MODEL_NAME=$MODEL_NAME\
#  PAD_TOKEN=$PAD_TOKEN\
#  OOV_TOKEN=$OOV_TOKEN\
#  BATCH_SIZE=$BATCH_SIZE\
#  VALIDATION_BATCH_SIZE=$BATCH_SIZE\
#  RANDOM_SEED=$RANDOM_SEED\
#  DROPOUT=$DROPOUT\
#  LR=$LR\
#  NUM_EPOCHS=$NUM_EPOCHS\
#  PRIOR_TYPE=$PRIOR_TYPE\
#  PRIOR_WEIGHT=$PRIOR_WEIGHT\
#  ENTITY_RATIO=$ENTITY_RATIO\
#  ENTITY_RATIO_MARGIN=$ENTITY_RATIO_MARGIN


# tacl-eer_eng-c_P-10000_rs2_eer
# reduced batch size by 1 because of CUDA OOM  
DATASET_LABEL=P-10000_rs$RANDOM_SEED
bash scripts/run_remote_ml_experiment.sh\
 PUBLIC_IP=3.236.52.210\
 PRIVATE_IP=172.31.60.222\
 RUN_JUPYTER=$RUN_JUPYTER\
 RUN_TENSORBOARD=$RUN_TENSORBOARD\
 LANG_LABEL=$LANG_LABEL\
 DATASET_LABEL=$DATASET_LABEL\
 METHOD_LABEL=$METHOD_LABEL\
 BASE_CONFIG=$BASE_CONFIG\
 ASSUME_COMPLETE=$ASSUME_COMPLETE\
 LANG_DIR=$LANG_DIR\
 TRAIN_DATA=entity.train-docs_$DATASET_LABEL.jsonl\
 DEV_DATA=$DEV_DATA\
 TEST_DATA=$TEST_DATA\
 VOCAB_PATH=$VOCAB_PATH\
 MODEL_NAME=$MODEL_NAME\
 PAD_TOKEN=$PAD_TOKEN\
 OOV_TOKEN=$OOV_TOKEN\
 BATCH_SIZE=14\
 VALIDATION_BATCH_SIZE=$BATCH_SIZE\
 RANDOM_SEED=$RANDOM_SEED\
 DROPOUT=$DROPOUT\
 LR=$LR\
 NUM_EPOCHS=$NUM_EPOCHS\
 PRIOR_TYPE=$PRIOR_TYPE\
 PRIOR_WEIGHT=$PRIOR_WEIGHT\
 ENTITY_RATIO=$ENTITY_RATIO\
 ENTITY_RATIO_MARGIN=$ENTITY_RATIO_MARGIN


# # tacl-eer_eng-c_gold_rs2_eer
# DATASET_LABEL=gold_rs$RANDOM_SEED
# bash scripts/run_remote_ml_experiment.sh\
#  PUBLIC_IP=3.230.142.167\
#  PRIVATE_IP=172.31.55.255\
#  RUN_JUPYTER=$RUN_JUPYTER\
#  RUN_TENSORBOARD=$RUN_TENSORBOARD\
#  LANG_LABEL=$LANG_LABEL\
#  DATASET_LABEL=$DATASET_LABEL\
#  METHOD_LABEL=$METHOD_LABEL\
#  BASE_CONFIG=experiments/supervised_tagger.jsonnet\
#  ASSUME_COMPLETE=$ASSUME_COMPLETE\
#  LANG_DIR=$LANG_DIR\
#  TRAIN_DATA=entity.train-docs.jsonl\
#  DEV_DATA=$DEV_DATA\
#  TEST_DATA=$TEST_DATA\
#  VOCAB_PATH=$VOCAB_PATH\
#  MODEL_NAME=$MODEL_NAME\
#  PAD_TOKEN=$PAD_TOKEN\
#  OOV_TOKEN=$OOV_TOKEN\
#  BATCH_SIZE=$BATCH_SIZE\
#  VALIDATION_BATCH_SIZE=$BATCH_SIZE\
#  RANDOM_SEED=$RANDOM_SEED\
#  DROPOUT=$DROPOUT\
#  LR=$LR\
#  NUM_EPOCHS=$NUM_EPOCHS\
#  PRIOR_TYPE=$PRIOR_TYPE\
#  PRIOR_WEIGHT=$PRIOR_WEIGHT\
#  ENTITY_RATIO=$ENTITY_RATIO\
#  ENTITY_RATIO_MARGIN=$ENTITY_RATIO_MARGIN
