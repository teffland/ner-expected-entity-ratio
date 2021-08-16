# Place the extracted ontonotes5 corpus in the ONTO_PATH path
ONTO_PATH="./data/ontonotes5"  # change this as needed

for LANG in english arabic chinese
do
    echo === Processing $LANG ===
    mkdir -p data/ontonotes5/processed/$LANG
    python -m ml.cmd.convert_ontonotes_data $ONTO_PATH/data/files/data/$LANG data/ontonotes5/processed/$LANG/\
    --as-sentences --is-complete

    mkdir -p data/ontonotes5/processed_docs/$LANG
    python -m ml.cmd.convert_ontonotes_data $ONTO_PATH/data/files/data/$LANG data/ontonotes5/processed_docs/$LANG/\
    --is-complete
done