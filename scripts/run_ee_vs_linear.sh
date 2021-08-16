# Comparing Linear, exhaustive annotation to EE annotation with same budget

RUN_JUPYTER=true
RUN_TENSORBOARD=true
LANG_LABEL=eng-c
LANG_DIR=data/conll2003/eng
DEV_DATA=entity.dev-docs.jsonl
TEST_DATA=entity.test-docs.jsonl
VOCAB_PATH=data/conll2003/roberta-entity.vocab
MODEL_NAME=roberta-base
PAD_TOKEN="<pad>"
OOV_TOKEN="<unk>"

BATCH_SIZE=15
RANDOM_SEED=0
DROPOUT=0.2
LR=2e-5
ENTITY_RATIO=0.15
ENTITY_RATIO_MARGIN=0.05


# Raw linear, only docs with annotations
# tacl-eer_eng-c_linear_raw-short
bash scripts/run_remote_ml_experiment.sh\
 PUBLIC_IP=3.239.202.240\
 PRIVATE_IP=172.31.9.217\
 RUN_JUPYTER=$RUN_JUPYTER\
 RUN_TENSORBOARD=$RUN_TENSORBOARD\
 LANG_LABEL=$LANG_LABEL\
 DATASET_LABEL=linear\
 METHOD_LABEL=raw-short\
 BASE_CONFIG=experiments/supervised_tagger_short.jsonnet\
 ASSUME_COMPLETE="true"\
 LANG_DIR=$LANG_DIR\
 TRAIN_DATA=entity.train-docs_P-1000_linear.jsonl\
 DEV_DATA=$DEV_DATA\
 TEST_DATA=$TEST_DATA\
 VOCAB_PATH=$VOCAB_PATH\
 MODEL_NAME=$MODEL_NAME\
 PAD_TOKEN=$PAD_TOKEN\
 OOV_TOKEN=$OOV_TOKEN\
 BATCH_SIZE=$BATCH_SIZE\
 VALIDATION_BATCH_SIZE=$BATCH_SIZE\
 RANDOM_SEED=$RANDOM_SEED\
 DROPOUT=$DROPOUT\
 LR=$LR\
 NUM_EPOCHS=50\
 PRIOR_TYPE=null\
 PRIOR_WEIGHT=0.0\
 ENTITY_RATIO=$ENTITY_RATIO\
 ENTITY_RATIO_MARGIN=$ENTITY_RATIO_MARGIN


# # EER EE, only docs with annotations
# # tacl-eer_eng-c_ee_eer-short
# bash scripts/run_remote_ml_experiment.sh\
#  PUBLIC_IP=3.228.1.36\
#  PRIVATE_IP=172.31.5.232\
#  RUN_JUPYTER=$RUN_JUPYTER\
#  RUN_TENSORBOARD=$RUN_TENSORBOARD\
#  LANG_LABEL=$LANG_LABEL\
#  DATASET_LABEL=ee\
#  METHOD_LABEL=eer-short\
#  BASE_CONFIG=experiments/supervised_tagger_short.jsonnet\
#  ASSUME_COMPLETE="false"\
#  LANG_DIR=$LANG_DIR\
#  TRAIN_DATA=entity.train-docs_P-1000.jsonl\
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
#  NUM_EPOCHS=30\
#  PRIOR_TYPE="eer-exact"\
#  PRIOR_WEIGHT=10.0\
#  ENTITY_RATIO=$ENTITY_RATIO\
#  ENTITY_RATIO_MARGIN=$ENTITY_RATIO_MARGIN


# # EER EE, all docs
# # Already happened at tacl-eer_eng-c_ee_eer