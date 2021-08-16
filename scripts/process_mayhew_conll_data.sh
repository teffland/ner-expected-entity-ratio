MAYHEW_PATH="./mayhew19"  # change this as needed

for LANG in eng deu esp ned
do
    echo === Processing $LANG ===
    mkdir -p data/conll2003/$LANG
    python -m ml.cmd.convert_mayhew_data $MAYHEW_PATH/data/$LANG/Train data/conll2003/$LANG/entity.train.jsonl\
    --as-sentences --is-complete
    python -m ml.cmd.convert_mayhew_data $MAYHEW_PATH/data/$LANG/Dev data/conll2003/$LANG/entity.dev.jsonl\
    --as-sentences --is-complete
    python -m ml.cmd.convert_mayhew_data $MAYHEW_PATH/data/$LANG/Test data/conll2003/$LANG/entity.test.jsonl\
    --as-sentences --is-complete

    python -m ml.cmd.convert_mayhew_data $MAYHEW_PATH/data/$LANG/Train data/conll2003/$LANG/entity.train-docs.jsonl\
    --is-complete
    python -m ml.cmd.convert_mayhew_data $MAYHEW_PATH/data/$LANG/Dev data/conll2003/$LANG/entity.dev-docs.jsonl\
    --is-complete
    python -m ml.cmd.convert_mayhew_data $MAYHEW_PATH/data/$LANG/Test data/conll2003/$LANG/entity.test-docs.jsonl\
    --is-complete
done