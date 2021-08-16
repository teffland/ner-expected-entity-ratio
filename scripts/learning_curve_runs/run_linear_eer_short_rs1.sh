# Learning curve experiment: EER on linear documents with annotations of varying coverage, rs1
RUN_JUPYTER=true
RUN_TENSORBOARD=true
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
RANDOM_SEED=1
DROPOUT=0.2
LR=2e-5
NUM_EPOCHS=50
PRIOR_TYPE="eer-exact"
PRIOR_WEIGHT=10.0
ENTITY_RATIO=0.15
ENTITY_RATIO_MARGIN=0.05


# tacl-eer_eng-c_P-100_linear_rs1_eer
DATASET_LABEL=P-100_linear_rs$RANDOM_SEED
bash scripts/run_remote_ml_experiment.sh\
 PUBLIC_IP=54.162.58.222\
 PRIVATE_IP=172.31.46.209\
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
 BATCH_SIZE=$BATCH_SIZE\
 VALIDATION_BATCH_SIZE=$BATCH_SIZE\
 RANDOM_SEED=$RANDOM_SEED\
 DROPOUT=$DROPOUT\
 LR=$LR\
 NUM_EPOCHS=$NUM_EPOCHS\
 PRIOR_TYPE=$PRIOR_TYPE\
 PRIOR_WEIGHT=$PRIOR_WEIGHT\
 ENTITY_RATIO=$ENTITY_RATIO\
 ENTITY_RATIO_MARGIN=$ENTITY_RATIO_MARGIN


# tacl-eer_eng-c_P-500_linear_rs1_eer
DATASET_LABEL=P-500_linear_rs$RANDOM_SEED
bash scripts/run_remote_ml_experiment.sh\
 PUBLIC_IP=18.232.75.192\
 PRIVATE_IP=172.31.33.245\
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
 BATCH_SIZE=$BATCH_SIZE\
 VALIDATION_BATCH_SIZE=$BATCH_SIZE\
 RANDOM_SEED=$RANDOM_SEED\
 DROPOUT=$DROPOUT\
 LR=$LR\
 NUM_EPOCHS=$NUM_EPOCHS\
 PRIOR_TYPE=$PRIOR_TYPE\
 PRIOR_WEIGHT=$PRIOR_WEIGHT\
 ENTITY_RATIO=$ENTITY_RATIO\
 ENTITY_RATIO_MARGIN=$ENTITY_RATIO_MARGIN


# tacl-eer_eng-c_P-1000_linear_rs1_eer
DATASET_LABEL=P-1000_linear_rs$RANDOM_SEED
bash scripts/run_remote_ml_experiment.sh\
 PUBLIC_IP=107.21.197.29\
 PRIVATE_IP=172.31.34.33\
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
 BATCH_SIZE=$BATCH_SIZE\
 VALIDATION_BATCH_SIZE=$BATCH_SIZE\
 RANDOM_SEED=$RANDOM_SEED\
 DROPOUT=$DROPOUT\
 LR=$LR\
 NUM_EPOCHS=$NUM_EPOCHS\
 PRIOR_TYPE=$PRIOR_TYPE\
 PRIOR_WEIGHT=$PRIOR_WEIGHT\
 ENTITY_RATIO=$ENTITY_RATIO\
 ENTITY_RATIO_MARGIN=$ENTITY_RATIO_MARGIN


# tacl-eer_eng-c_P-5000_linear_rs1_eer
DATASET_LABEL=P-5000_linear_rs$RANDOM_SEED
bash scripts/run_remote_ml_experiment.sh\
 PUBLIC_IP=3.80.184.26\
 PRIVATE_IP=172.31.39.22\
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
 BATCH_SIZE=$BATCH_SIZE\
 VALIDATION_BATCH_SIZE=$BATCH_SIZE\
 RANDOM_SEED=$RANDOM_SEED\
 DROPOUT=$DROPOUT\
 LR=$LR\
 NUM_EPOCHS=$NUM_EPOCHS\
 PRIOR_TYPE=$PRIOR_TYPE\
 PRIOR_WEIGHT=$PRIOR_WEIGHT\
 ENTITY_RATIO=$ENTITY_RATIO\
 ENTITY_RATIO_MARGIN=$ENTITY_RATIO_MARGIN


# tacl-eer_eng-c_P-10000_linear_rs1_eer
DATASET_LABEL=P-10000_linear_rs$RANDOM_SEED
bash scripts/run_remote_ml_experiment.sh\
 PUBLIC_IP=54.86.132.245\
 PRIVATE_IP=172.31.47.25\
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
 BATCH_SIZE=$BATCH_SIZE\
 VALIDATION_BATCH_SIZE=$BATCH_SIZE\
 RANDOM_SEED=$RANDOM_SEED\
 DROPOUT=$DROPOUT\
 LR=$LR\
 NUM_EPOCHS=$NUM_EPOCHS\
 PRIOR_TYPE=$PRIOR_TYPE\
 PRIOR_WEIGHT=$PRIOR_WEIGHT\
 ENTITY_RATIO=$ENTITY_RATIO\
 ENTITY_RATIO_MARGIN=$ENTITY_RATIO_MARGIN
