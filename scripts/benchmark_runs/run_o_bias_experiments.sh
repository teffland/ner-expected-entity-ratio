# Run experiments for the Raw+CD baseline

#### NNS
# Conll english
echo logs/21-06-01_eng-c_nns_obias.out
python -m ml.cmd.o_bias_experiment\
  --model-folder experiments/eng-c/nns/raw-mle\
  --dev-file data/conll2003/eng/entity.dev-docs.jsonl\
  --test-file data/conll2003/eng/entity.test-docs.jsonl\
  --n-calls 30 &> logs/21-06-01_eng-c_nns_obias.out

# Conll german
echo logs/21-06-01_deu_nns_obias.out
python -m ml.cmd.o_bias_experiment\
  --model-folder experiments/deu/nns/raw-mle\
  --dev-file data/conll2003/deu/entity.dev-docs.jsonl\
  --test-file data/conll2003/deu/entity.test-docs.jsonl\
  --n-calls 30 &> logs/21-06-01_deu_nns_obias.out

# Conll spanish
echo logs/21-06-01_esp_nns_obias.out
python -m ml.cmd.o_bias_experiment\
  --model-folder experiments/esp/nns/raw-mle\
  --dev-file data/conll2003/esp/entity.dev-docs.jsonl\
  --test-file data/conll2003/esp/entity.test-docs.jsonl\
  --n-calls 30 &> logs/21-06-01_esp_nns_obias.out

# Conll dutch
echo logs/21-06-01_ned_nns_obias.out
python -m ml.cmd.o_bias_experiment\
  --model-folder experiments/ned/nns/raw-mle\
  --dev-file data/conll2003/ned/entity.dev-docs.jsonl\
  --test-file data/conll2003/ned/entity.test-docs.jsonl\
  --n-calls 30 &> logs/21-06-01_ned_nns_obias.out

# ontonotes english
echo logs/21-06-01_eng-o_nns_obias.out
python -m ml.cmd.o_bias_experiment\
  --model-folder experiments/eng-o/nns/raw-mle\
  --dev-file data/ontonotes5/processed_docs/english/dev.jsonl\
  --test-file data/ontonotes5/processed_docs/english/test.jsonl\
  --n-calls 30 &> logs/21-06-01_eng-o_nns_obias.out

# ontonotes chinese
echo logs/21-06-01_chi_nns_obias.out
python -m ml.cmd.o_bias_experiment\
  --model-folder experiments/chi/nns/raw-mle/raw-mle\
  --dev-file data/ontonotes5/processed_docs/chinese/dev.jsonl\
  --test-file data/ontonotes5/processed_docs/chinese/test.jsonl\
  --n-calls 30 &> logs/21-06-01_chi_nns_obias.out

# ontonotes arabic
echo logs/21-06-01_ara_nns_obias.out
python -m ml.cmd.o_bias_experiment\
  --model-folder experiments/ara/nns/raw-mle/raw-mle\
  --dev-file data/ontonotes5/processed_docs/arabic/dev.jsonl\
  --test-file data/ontonotes5/processed_docs/arabic/test.jsonl\
  --n-calls 30 &> logs/21-06-01_ara_nns_obias.out








# #### EE
# Conll english
echo logs/21-06-01_eng-c_ee_obias.out
python -m ml.cmd.o_bias_experiment\
  --model-folder experiments/eng-c/ee/raw-short-sent\
  --dev-file data/conll2003/eng/entity.dev-docs.jsonl\
  --test-file data/conll2003/eng/entity.test-docs.jsonl\
  --n-calls 30 &> logs/21-06-01_eng-c_ee_obias.out

# Conll german
echo logs/21-06-01_deu_ee_obias.out
python -m ml.cmd.o_bias_experiment\
  --model-folder experiments/deu/ee/raw-short-sent\
  --dev-file data/conll2003/deu/entity.dev-docs.jsonl\
  --test-file data/conll2003/deu/entity.test-docs.jsonl\
  --n-calls 30 &> logs/21-06-01_deu_ee_obias.out

# Conll spanish
echo logs/21-06-01_esp_ee_obias.out
python -m ml.cmd.o_bias_experiment\
  --model-folder experiments/esp/ee/raw-short-sent\
  --dev-file data/conll2003/esp/entity.dev-docs.jsonl\
  --test-file data/conll2003/esp/entity.test-docs.jsonl\
  --n-calls 30 &> logs/21-06-01_esp_ee_obias.out

# Conll dutch
echo logs/21-06-01_ned_ee_obias.out
python -m ml.cmd.o_bias_experiment\
  --model-folder experiments/ned/ee/raw-short-sent\
  --dev-file data/conll2003/ned/entity.dev-docs.jsonl\
  --test-file data/conll2003/ned/entity.test-docs.jsonl\
  --n-calls 30 &> logs/21-06-01_ned_ee_obias.out

# ontonotes english
echo logs/21-06-01_eng-o_ee_obias.out
python -m ml.cmd.o_bias_experiment\
  --model-folder experiments/eng-o/ee/raw-short-sent\
  --dev-file data/ontonotes5/processed_docs/english/dev.jsonl\
  --test-file data/ontonotes5/processed_docs/english/test.jsonl\
  --n-calls 30 &> logs/21-06-01_eng-o_ee_obias.out

# ontonotes chinese
echo logs/21-06-01_chi_ee_obias.out
python -m ml.cmd.o_bias_experiment\
  --model-folder experiments/chi/ee/raw-short-sent\
  --dev-file data/ontonotes5/processed_docs/chinese/dev.jsonl\
  --test-file data/ontonotes5/processed_docs/chinese/test.jsonl\
  --n-calls 30 &> logs/21-06-01_chi_ee_obias.out

# ontonotes arabic
echo logs/21-06-01_ara_ee_obias.out
python -m ml.cmd.o_bias_experiment\
  --model-folder experiments/ara/ee/raw-short-sent\
  --dev-file data/ontonotes5/processed_docs/arabic/dev.jsonl\
  --test-file data/ontonotes5/processed_docs/arabic/test.jsonl\
  --n-calls 30 &> logs/21-06-01_ara_ee_obias.out