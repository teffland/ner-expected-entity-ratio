# Benchmark Experiment: BCL-BERT on EE doc data, dropping sentences without annotations (the adapted Mayhew 19 code)
RUN_JUPYTER=false
RUN_TENSORBOARD=false
DATASET_LABEL=ee-short-sent
METHOD_LABEL=bcl-bert
TRAIN_SUFFIX=_P-1000


# # Conll english
# # tacl-eer_eng-c_ee-short-sent_bcl-bert
# bash scripts/run_remote_mayhew19_experiment.sh\
#  PUBLIC_IP=3.235.140.93\
#  PRIVATE_IP=172.31.54.102\
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
# # tacl-eer_deu_ee-short-sent_bcl-bert
# bash scripts/run_remote_mayhew19_experiment.sh\
#  PUBLIC_IP=3.238.134.226\
#  PRIVATE_IP=172.31.55.246\
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
# tacl-eer_esp_ee-short-sent_bcl-bert
bash scripts/run_remote_mayhew19_experiment.sh\
 PUBLIC_IP=3.235.143.253\
 PRIVATE_IP=172.31.61.93\
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
# # tacl-eer_ned_ee-short-sent_bcl-bert
# bash scripts/run_remote_mayhew19_experiment.sh\
#  PUBLIC_IP=3.238.112.197\
#  PRIVATE_IP=172.31.56.97\
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
# # tacl-eer_eng-o_ee-short-sent_bcl-bert
# bash scripts/run_remote_mayhew19_experiment.sh\
#  PUBLIC_IP=18.232.49.13\
#  PRIVATE_IP=172.31.58.142\
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
# # tacl-eer_chi_ee-short-sent_bcl-bert
# bash scripts/run_remote_mayhew19_experiment.sh\
#  PUBLIC_IP=34.231.70.67\
#  PRIVATE_IP=172.31.48.146\
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


# # Ontonotes arabic
# # tacl-eer_ara_ee-short-sent_bcl-bert
# bash scripts/run_remote_mayhew19_experiment.sh\
#  PUBLIC_IP=34.232.64.45\
#  PRIVATE_IP=172.31.54.53\
#  RUN_JUPYTER=$RUN_JUPYTER\
#  RUN_TENSORBOARD=$RUN_TENSORBOARD\
#  LANG_LABEL=ara\
#  DATASET_LABEL=$DATASET_LABEL\
#  METHOD_LABEL=$METHOD_LABEL\
#  LANG_DIR=data/ontonotes5/processed_docs/arabic\
#  TRAIN_DATA=train$TRAIN_SUFFIX.jsonl\
#  DEV_DATA=dev.jsonl\
#  TEST_DATA=test.jsonl\
#  VOCAB_PATH=data/ontonotes5/processed_docs/mayhew-vocab\
#  BINARY_VOCAB_PATH=data/ontonotes5/processed_docs/mayhew-binary-vocab\
#  VECTORS_PATH=data/vectors/fasttext.ara.300.vec
 

