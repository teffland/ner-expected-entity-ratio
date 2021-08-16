# Precompute simulated datasets (commented out are already done in the repo)

# Benchmarks
# for LANG in eng esp deu ned
# do
#     echo Conll $LANG
#     python -m ml.cmd.subsample_data_mayhew data/conll2003/$LANG/entity.train-docs.jsonl --recall 0.5 --precision 0.9
#     python -m ml.cmd.subsample_data_ours data/conll2003/$LANG/entity.train-docs.jsonl --n 1000 --n-per-doc 10 --drop-prob 0.2
# done

for LANG in english chinese arabic
do
    echo Ontonotes $LANG
    python -m ml.cmd.subsample_data_mayhew data/ontonotes5/processed_docs/$LANG/train.jsonl --recall 0.5 --precision 0.9
    python -m ml.cmd.subsample_data_ours data/ontonotes5/processed_docs/$LANG/train.jsonl --n 1000 --n-per-doc 10 --drop-prob 0.2
done

# Learning curves
# for R in 0 1 2
# do
#     for M in 100 500 1000 5000 10000
#     do
#         echo Conll english $R $M
#         if [ $M == 10000 ]
#         then
#             NPER=15
#         else
#             NPER=10
#         fi
#         python -m ml.cmd.subsample_data_ours data/conll2003/eng/entity.train-docs.jsonl --random-seed $R --n $M --n-per-doc $NPER --drop-prob 0.2
#         python -m ml.cmd.subsample_data_ours data/conll2003/eng/entity.train-docs.jsonl --random-seed $R --n $M --n-per-doc 0 --drop-prob 0.0 --suffix "_linear"
#     done
# done


