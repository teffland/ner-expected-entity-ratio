# Exactly the same as supervised_tagger.jsonnet, except the train dataset reader will drop
# any unannotated sentences after the last annotated sentence in a document
# -- it was easier to do this as a separate config
# than modify all of the preexisting scripts.

# Metadata
local strContains(s, substr) =  std.length(std.findSubstr(substr, s)) > 0;
local parseBool(x) = if x == "true" then true else false;
local assume_complete = parseBool(std.extVar('ASSUME_COMPLETE'));
local train_data_path = std.extVar('TRAIN_DATA_PATH');
local dev_data_path = std.extVar('DEV_DATA_PATH');
local test_data_path = std.extVar('TEST_DATA_PATH');
local vocab_path = std.extVar("VOCAB_PATH");
local model_name = std.extVar('MODEL_NAME');
local model_dim = 768;
local pad_token = std.extVar('PAD_TOKEN');
local oov_token = std.extVar('OOV_TOKEN');

# Hps
local batch_size = std.parseJson(std.extVar('BATCH_SIZE'));
local validation_batch_size = std.parseJson(std.extVar('VALIDATION_BATCH_SIZE'));
local random_seed = std.parseJson(std.extVar('RANDOM_SEED'));
local dropout = std.parseJson(std.extVar('DROPOUT'));
local lr = std.parseJson(std.extVar('LR'));
local num_epochs = std.parseJson(std.extVar('NUM_EPOCHS'));
local prior_loss_type = std.extVar('PRIOR_TYPE');
local prior_weight = std.parseJson(std.extVar('PRIOR_WEIGHT'));
local entity_ratio = std.parseJson(std.extVar('ENTITY_RATIO'));
local entity_ratio_margin = std.parseJson(std.extVar('ENTITY_RATIO_MARGIN'));

# Very small datasets run over dataset multiple times per "epoch"
local batches_per_epoch = (
  if strContains(train_data_path, '_P-100_') 
  || strContains(train_data_path, '_P-500_')
  || strContains(train_data_path, '_P-1000_') 
  then 10 
  else null
);
{
  random_seed:random_seed,
  numpy_seed:random_seed,
  pytorch_seed:random_seed,
  train_data_path: train_data_path,
  validation_data_path: dev_data_path,
  test_data_path: test_data_path,
  evaluate_on_test: true,
  vocabulary: {
    type: "from_files",
    directory: vocab_path,
    padding_token: pad_token,
    oov_token: oov_token,
  },
  dataset_reader: {
    type: "partial-jsonl",
    model_name: model_name,
    assume_complete: assume_complete,
    drop_unannotated_sentences: true,
    kind: "entity",
    lazy: false,
  },
  validation_dataset_reader: {
    type: "partial-jsonl",
    model_name: model_name,
    assume_complete: true,
    kind: "entity",
    lazy: false,
  },
  data_loader: {
    batch_sampler: {
      type: "bucket",
      batch_size: batch_size,
      sorting_keys: ["tokens"],
    },
    batches_per_epoch: batches_per_epoch,
  },
  validation_data_loader: {
    batch_sampler: {
      type: "bucket",
      batch_size: validation_batch_size,
      sorting_keys: ["tokens"],
    }
  },
  model: {
    type: "partial-supervised-tagger",
    text_field_embedder: {
      token_embedders: {
        tokens: {
          type: "pretrained_transformer",
          model_name: model_name,
          train_parameters: true,
        },
      }
    },
    dropout: dropout,
    prior_loss_type: prior_loss_type,
    prior_loss_weight: prior_weight,
    entity_ratio: entity_ratio,
    entity_ratio_margin: entity_ratio_margin,
  },
  trainer: {
    num_epochs: num_epochs,
    patience: 20,
    cuda_device: 0,
    grad_clipping: 5.0,
    checkpointer: {
      num_serialized_models_to_keep: 1,
    },
    tensorboard_writer: {
      summary_interval: 1,
      should_log_learning_rate: true,
    },
    validation_metric: "+f1-measure-overall",
    optimizer: {
      type: "adam",
      lr: lr,
      weight_decay: 0.0,
    },
    learning_rate_scheduler: {
      type: "slanted_triangular",
      num_epochs: num_epochs,
      cut_frac: 0.1,
      ratio: 16,
    },    
  }
}
