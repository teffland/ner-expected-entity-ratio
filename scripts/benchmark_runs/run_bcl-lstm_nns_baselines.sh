# Benchmark Experiment: BCL-LSTM on NNS sentence data (the Mayhew 19 code)
RUN_JUPYTER=false
RUN_TENSORBOARD=false
DATASET_LABEL=nns
METHOD_LABEL=bcl-lstm
TRAIN_SUFFIX=_r0.5_p0.9


# # Conll english
# # tacl-eer_eng-c_nns_bcl-lstm
# bash scripts/run_remote_mayhew19_experiment.sh\
#  PUBLIC_IP=3.239.79.140\
#  PRIVATE_IP=172.31.7.22\
#  RUN_JUPYTER=$RUN_JUPYTER\
#  RUN_TENSORBOARD=$RUN_TENSORBOARD\
#  LANG_LABEL=eng-c\
#  DATASET_LABEL=$DATASET_LABEL\
#  METHOD_LABEL=$METHOD_LABEL\
#  LANG_DIR=data/conll2003/eng\
#  TRAIN_DATA=entity.train$TRAIN_SUFFIX.jsonl\
#  DEV_DATA=entity.dev.jsonl\
#  TEST_DATA=entity.test.jsonl\
#  VOCAB_PATH=data/conll2003/mayhew-entity.vocab\
#  BINARY_VOCAB_PATH=data/conll2003/mayhew-binary-entity.vocab\
#  VECTORS_PATH=data/vectors/glove.6B.50d.txtå


# # Conll german
# # tacl-eer_deu_nns_bcl-lstm
# bash scripts/run_remote_mayhew19_experiment.sh\
#  PUBLIC_IP=3.238.228.127\
#  PRIVATE_IP=172.31.50.94\
#  RUN_JUPYTER=$RUN_JUPYTER\
#  RUN_TENSORBOARD=$RUN_TENSORBOARD\
#  LANG_LABEL=deu\
#  DATASET_LABEL=$DATASET_LABEL\
#  METHOD_LABEL=$METHOD_LABEL\
#  LANG_DIR=data/conll2003/deu\
#  TRAIN_DATA=entity.train$TRAIN_SUFFIX.jsonl\
#  DEV_DATA=entity.dev.jsonl\
#  TEST_DATA=entity.test.jsonl\
#  VOCAB_PATH=data/conll2003/mayhew-entity.vocab\
#  BINARY_VOCAB_PATH=data/conll2003/mayhew-binary-entity.vocab\
#  VECTORS_PATH=data/vectors/fasttext.deu.300.vec


# # Conll spanish
# # tacl-eer_esp_nns_bcl-lstm
# bash scripts/run_remote_mayhew19_experiment.sh\
#  PUBLIC_IP=3.236.119.143\
#  PRIVATE_IP=172.31.62.193\
#  RUN_JUPYTER=$RUN_JUPYTER\
#  RUN_TENSORBOARD=$RUN_TENSORBOARD\
#  LANG_LABEL=esp\
#  DATASET_LABEL=$DATASET_LABEL\
#  METHOD_LABEL=$METHOD_LABEL\
#  LANG_DIR=data/conll2003/esp\
#  TRAIN_DATA=entity.train$TRAIN_SUFFIX.jsonl\
#  DEV_DATA=entity.dev.jsonl\
#  TEST_DATA=entity.test.jsonl\
#  VOCAB_PATH=data/conll2003/mayhew-entity.vocab\
#  BINARY_VOCAB_PATH=data/conll2003/mayhew-binary-entity.vocab\
#  VECTORS_PATH=data/vectors/fasttext.esp.300.vec


# # Conll dutch
# # tacl-eer_ned_nns_bcl-lstm
# bash scripts/run_remote_mayhew19_experiment.sh\
#  PUBLIC_IP=18.207.134.138\
#  PRIVATE_IP=172.31.58.25\
#  RUN_JUPYTER=$RUN_JUPYTER\
#  RUN_TENSORBOARD=$RUN_TENSORBOARD\
#  LANG_LABEL=ned\
#  DATASET_LABEL=$DATASET_LABEL\
#  METHOD_LABEL=$METHOD_LABEL\
#  LANG_DIR=data/conll2003/ned\
#  TRAIN_DATA=entity.train$TRAIN_SUFFIX.jsonl\
#  DEV_DATA=entity.dev.jsonl\
#  TEST_DATA=entity.test.jsonl\
#  VOCAB_PATH=data/conll2003/mayhew-entity.vocab\
#  BINARY_VOCAB_PATH=data/conll2003/mayhew-binary-entity.vocab\
#  VECTORS_PATH=data/vectors/fasttext.ned.300.vec


# Ontonotes english
# tacl-eer_eng-o_nns_bcl-lstm
bash scripts/run_remote_mayhew19_experiment.sh\
 PUBLIC_IP=3.239.124.108\
 PRIVATE_IP=172.31.56.61\
 RUN_JUPYTER=$RUN_JUPYTER\
 RUN_TENSORBOARD=$RUN_TENSORBOARD\
 LANG_LABEL=eng-o\
 DATASET_LABEL=$DATASET_LABEL\
 METHOD_LABEL=$METHOD_LABEL\
 LANG_DIR=data/ontonotes5/processed/english\
 TRAIN_DATA=train$TRAIN_SUFFIX.jsonl\
 DEV_DATA=dev.jsonl\
 TEST_DATA=test.jsonl\
 VOCAB_PATH=data/ontonotes5/processed/mayhew-vocab\
 BINARY_VOCAB_PATH=data/ontonotes5/processed/mayhew-binary-vocab\
 VECTORS_PATH=data/vectors/glove.6B.50d.txt


# # Ontonotes chinese
# # tacl-eer_chi_nns_bcl-lstm
# bash scripts/run_remote_mayhew19_experiment.sh\
#  PUBLIC_IP=3.238.110.102\
#  PRIVATE_IP=172.31.58.47\
#  RUN_JUPYTER=$RUN_JUPYTER\
#  RUN_TENSORBOARD=$RUN_TENSORBOARD\
#  LANG_LABEL=chi\
#  DATASET_LABEL=$DATASET_LABEL\
#  METHOD_LABEL=$METHOD_LABEL\
#  LANG_DIR=data/ontonotes5/processed/chinese\
#  TRAIN_DATA=train$TRAIN_SUFFIX.jsonl\
#  DEV_DATA=dev.jsonl\
#  TEST_DATA=test.jsonl\
#  VOCAB_PATH=data/ontonotes5/processed/mayhew-vocab\
#  BINARY_VOCAB_PATH=data/ontonotes5/processed/mayhew-binary-vocab\
#  VECTORS_PATH=data/vectors/fasttext.chi.300.vec


# # Ontonotes arabic
# # tacl-eer_ara_nns_bcl-lstm
# bash scripts/run_remote_mayhew19_experiment.sh\
#  PUBLIC_IP=3.238.201.116\
#  PRIVATE_IP=172.31.50.98\
#  RUN_JUPYTER=$RUN_JUPYTER\
#  RUN_TENSORBOARD=$RUN_TENSORBOARD\
#  LANG_LABEL=ara\
#  DATASET_LABEL=$DATASET_LABEL\
#  METHOD_LABEL=$METHOD_LABEL\
#  LANG_DIR=data/ontonotes5/processed/arabic\
#  TRAIN_DATA=train$TRAIN_SUFFIX.jsonl\
#  DEV_DATA=dev.jsonl\
#  TEST_DATA=test.jsonl\
#  VOCAB_PATH=data/ontonotes5/processed/mayhew-vocab\
#  BINARY_VOCAB_PATH=data/ontonotes5/processed/mayhew-binary-vocab\
#  VECTORS_PATH=data/vectors/fasttext.ara.300.vec
 

