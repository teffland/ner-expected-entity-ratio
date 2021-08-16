# Find max batch sizes
export ASSUME_COMPLETE=false
export BATCH_SIZE=1
export RANDOM_SEED=0
export DROPOUT=0.2
export LR=2e-5
export NUM_EPOCHS=20
export PRIOR_TYPE="eer-exact"
export PRIOR_WEIGHT=1.0
export ENTITY_RATIO=0.15
export ENTITY_RATIO_MARGIN=0.5

echo CONLL
for LANG in eng esp deu ned
do
    LANG_DIR=data/conll2003/$LANG
    export TRAIN_DATA_PATH=$LANG_DIR/entity.train-docs.jsonl
    export DEV_DATA_PATH=$LANG_DIR/entity.dev-docs.jsonl
    export TEST_DATA_PATH=$LANG_DIR/entity.test-docs.jsonl
    if [ $LANG = 'eng' ]
    then
        export VOCAB_PATH=data/conll2003/roberta-entity.vocab
        export MODEL_NAME=roberta-base
        export PAD_TOKEN="<pad>"
        export OOV_TOKEN="<unk>"
    else
        export VOCAB_PATH=data/conll2003/mbert-entity.vocab
        export MODEL_NAME=bert-base-multilingual-cased
        export PAD_TOKEN="[PAD]"
        export OOV_TOKEN="[UNK]"
    fi

    SERIALIZATION_DIR=experiments/find_batch_size/conll2003-$LANG
    BASE_CONFIG=experiments/supervised_tagger.jsonnet
    LOG=logs/find_batch_size_conll2003-$LANG.log
    echo RUNNING CONLL LANG: $LANG
    echo Check the log at: $LOG
    /home/ubuntu/anaconda3/envs/env/bin/allennlp train $BASE_CONFIG\
        -f -s $SERIALIZATION_DIR\
        -o '{"data_loader": {"batch_sampler": {"type":"find_bucket_max_batch_size"}}}'\
        --include ml\
        &> $LOG
    tail -n75 $LOG
done


echo ONTONOTES
for LANG in english chinese arabic
do
    LANG_DIR=data/ontonotes5/processed_docs/$LANG
    export TRAIN_DATA_PATH=$LANG_DIR/train.jsonl
    export DEV_DATA_PATH=$LANG_DIR/dev.jsonl
    export TEST_DATA_PATH=$LANG_DIR/test.jsonl
    if [ $LANG = 'english' ]
    then
        export VOCAB_PATH=data/ontonotes5/processed/roberta-entity.vocab
        export MODEL_NAME=roberta-base
        export PAD_TOKEN="<pad>"
        export OOV_TOKEN="<unk>"
    else
        export VOCAB_PATH=data/ontonotes5/processed/mbert-entity.vocab
        export MODEL_NAME=bert-base-multilingual-cased
        export PAD_TOKEN="[PAD]"
        export OOV_TOKEN="[UNK]"
    fi

    SERIALIZATION_DIR=experiments/find_batch_size/ontonotes5-$LANG
    BASE_CONFIG=experiments/supervised_tagger.jsonnet
    LOG=logs/find_batch_size_ontonotes5-$LANG.log
    echo RUNNING ONTO LANG: $LANG
    echo Check the log at: $LOG
    /home/ubuntu/anaconda3/envs/env/bin/allennlp train $BASE_CONFIG\
        -f -s $SERIALIZATION_DIR\
        -o '{"data_loader": {"batch_sampler": {"type":"find_bucket_max_batch_size"}}}'\
        --include ml\
        &> $LOG
    tail -n75 $LOG
done