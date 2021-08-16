# Benchmark Experiment: BCL-BERT on NNS doc data (the adapted Mayhew 19 code)
RUN_JUPYTER=true
RUN_TENSORBOARD=true
DATASET_LABEL=nns
METHOD_LABEL=bcl-bert
TRAIN_SUFFIX=_r0.5_p0.9


# # Conll english
# # tacl-eer_eng-c_nns_bcl-bert
# bash scripts/run_remote_mayhew19_experiment.sh\
#  PUBLIC_IP=54.198.34.181\
#  PRIVATE_IP=172.31.34.26\
#  RUN_JUPYTER=$RUN_JUPYTER\
#  RUN_TENSORBOARD=$RUN_TENSORBOARD\
#  LANG_LABEL=eng-c\
#  DATASET_LABEL=$DATASET_LABEL\
#  METHOD_LABEL=$METHOD_LABEL\
#  LANG_DIR=data/conll2003/eng\
#  TRAIN_DATA=entity.train-docs$TRAIN_SUFFIX.jsonl\
#  DEV_DATA=entity.dev-docs.jsonl\
#  TEST_DATA=entity.test-docs.jsonl\
#  VOCAB_PATH=data/conll2003/mayhew-entity.vocab\
#  BINARY_VOCAB_PATH=data/conll2003/mayhew-binary-entity.vocab\
#  VECTORS_PATH=data/vectors/glove.6B.50d.txt


# # Conll german
# # tacl-eer_deu_nns_bcl-bert
# bash scripts/run_remote_mayhew19_experiment.sh\
#  PUBLIC_IP=18.206.86.190\
#  PRIVATE_IP=172.31.1.31\
#  RUN_JUPYTER=$RUN_JUPYTER\
#  RUN_TENSORBOARD=$RUN_TENSORBOARD\
#  LANG_LABEL=deu\
#  DATASET_LABEL=$DATASET_LABEL\
#  METHOD_LABEL=$METHOD_LABEL\
#  LANG_DIR=data/conll2003/deu\
#  TRAIN_DATA=entity.train-docs$TRAIN_SUFFIX.jsonl\
#  DEV_DATA=entity.dev-docs.jsonl\
#  TEST_DATA=entity.test-docs.jsonl\
#  VOCAB_PATH=data/conll2003/mayhew-entity.vocab\
#  BINARY_VOCAB_PATH=data/conll2003/mayhew-binary-entity.vocab\
#  VECTORS_PATH=data/vectors/fasttext.deu.300.vec


# Conll spanish
# tacl-eer_esp_nns_bcl-bert
bash scripts/run_remote_mayhew19_experiment.sh\
 PUBLIC_IP=3.238.34.224\
 PRIVATE_IP=172.31.7.200\
 RUN_JUPYTER=$RUN_JUPYTER\
 RUN_TENSORBOARD=$RUN_TENSORBOARD\
 LANG_LABEL=esp\
 DATASET_LABEL=$DATASET_LABEL\
 METHOD_LABEL=$METHOD_LABEL\
 LANG_DIR=data/conll2003/esp\
 TRAIN_DATA=entity.train-docs$TRAIN_SUFFIX.jsonl\
 DEV_DATA=entity.dev-docs.jsonl\
 TEST_DATA=entity.test-docs.jsonl\
 VOCAB_PATH=data/conll2003/mayhew-entity.vocab\
 BINARY_VOCAB_PATH=data/conll2003/mayhew-binary-entity.vocab\
 VECTORS_PATH=data/vectors/fasttext.esp.300.vec


# # Conll dutch
# # tacl-eer_ned_nns_bcl-bert
# bash scripts/run_remote_mayhew19_experiment.sh\
#  PUBLIC_IP=18.210.7.141\
#  PRIVATE_IP=172.31.9.131\
#  RUN_JUPYTER=$RUN_JUPYTER\
#  RUN_TENSORBOARD=$RUN_TENSORBOARD\
#  LANG_LABEL=ned\
#  DATASET_LABEL=$DATASET_LABEL\
#  METHOD_LABEL=$METHOD_LABEL\
#  LANG_DIR=data/conll2003/ned\
#  TRAIN_DATA=entity.train-docs$TRAIN_SUFFIX.jsonl\
#  DEV_DATA=entity.dev-docs.jsonl\
#  TEST_DATA=entity.test-docs.jsonl\
#  VOCAB_PATH=data/conll2003/mayhew-entity.vocab\
#  BINARY_VOCAB_PATH=data/conll2003/mayhew-binary-entity.vocab\
#  VECTORS_PATH=data/vectors/fasttext.ned.300.vec


# # Ontonotes english
# # tacl-eer_eng-o_nns_bcl-bert
# bash scripts/run_remote_mayhew19_experiment.sh\
#  PUBLIC_IP=3.237.95.43\
#  PRIVATE_IP=172.31.7.82\
#  RUN_JUPYTER=$RUN_JUPYTER\
#  RUN_TENSORBOARD=$RUN_TENSORBOARD\
#  LANG_LABEL=eng-o\
#  DATASET_LABEL=$DATASET_LABEL\
#  METHOD_LABEL=$METHOD_LABEL\
#  LANG_DIR=data/ontonotes5/processed_docs/english\
#  TRAIN_DATA=train$TRAIN_SUFFIX.jsonl\
#  DEV_DATA=dev.jsonl\
#  TEST_DATA=test.jsonl\
#  VOCAB_PATH=data/ontonotes5/processed_docs/mayhew-vocab\
#  BINARY_VOCAB_PATH=data/ontonotes5/processed_docs/mayhew-binary-vocab\
#  VECTORS_PATH=data/vectors/glove.6B.50d.txt


# # Ontonotes chinese
# # tacl-eer_chi_nns_bcl-bert
# bash scripts/run_remote_mayhew19_experiment.sh\
#  PUBLIC_IP=3.239.107.166\
#  PRIVATE_IP=172.31.2.158\
#  RUN_JUPYTER=$RUN_JUPYTER\
#  RUN_TENSORBOARD=$RUN_TENSORBOARD\
#  LANG_LABEL=chi\
#  DATASET_LABEL=$DATASET_LABEL\
#  METHOD_LABEL=$METHOD_LABEL\
#  LANG_DIR=data/ontonotes5/processed_docs/chinese\
#  TRAIN_DATA=train$TRAIN_SUFFIX.jsonl\
#  DEV_DATA=dev.jsonl\
#  TEST_DATA=test.jsonl\
#  VOCAB_PATH=data/ontonotes5/processed_docs/mayhew-vocab\
#  BINARY_VOCAB_PATH=data/ontonotes5/processed_docs/mayhew-binary-vocab\
#  VECTORS_PATH=data/vectors/fasttext.chi.300.vec


# Ontonotes arabic
# tacl-eer_ara_nns_bcl-bert
bash scripts/run_remote_mayhew19_experiment.sh\
 PUBLIC_IP=3.237.95.118\
 PRIVATE_IP=172.31.9.104\
 RUN_JUPYTER=$RUN_JUPYTER\
 RUN_TENSORBOARD=$RUN_TENSORBOARD\
 LANG_LABEL=ara\
 DATASET_LABEL=$DATASET_LABEL\
 METHOD_LABEL=$METHOD_LABEL\
 LANG_DIR=data/ontonotes5/processed_docs/arabic\
 TRAIN_DATA=train$TRAIN_SUFFIX.jsonl\
 DEV_DATA=dev.jsonl\
 TEST_DATA=test.jsonl\
 VOCAB_PATH=data/ontonotes5/processed_docs/mayhew-vocab\
 BINARY_VOCAB_PATH=data/ontonotes5/processed_docs/mayhew-binary-vocab\
 VECTORS_PATH=data/vectors/fasttext.ara.300.vec
 

