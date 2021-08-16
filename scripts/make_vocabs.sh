# Precompute vocab files for models so we can control which tags are which index (and for speed)

# python -m ml.cmd.make_vocab roberta-base data/conll2003/eng/entity.train.jsonl data/conll2003/roberta-entity.vocab
# python -m ml.cmd.make_vocab bert-base-multilingual-cased data/conll2003/deu/entity.train.jsonl data/conll2003/mbert-entity.vocab
python -m ml.cmd.make_vocab roberta-base data/ontonotes5/processed/english/train.jsonl data/ontonotes5/processed/roberta-entity.vocab
python -m ml.cmd.make_vocab bert-base-multilingual-cased data/ontonotes5/processed/chinese/train.jsonl data/ontonotes5/processed/mbert-entity.vocab