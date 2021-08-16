## Everything needed to run an experiment on its own aws instance

# Parse keyword arguments into environment variables
# derived from https://unix.stackexchange.com/questions/129391/passing-named-arguments-to-shell-scripts
for ARGUMENT in "$@"
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)   
    case "$KEY" in
            IS_REMOTE)             IS_REMOTE=${VALUE} ;;
            RUN_JUPYTER)           RUN_JUPYTER=${VALUE} ;;
            RUN_TENSORBOARD)       RUN_TENSORBOARD=${VALUE} ;;
            LANG_LABEL)            LANG_LABEL=${VALUE} ;;  # eg. eng-o
            DATASET_LABEL)         DATASET_LABEL=${VALUE} ;;  # eg. mayhew or ours
            METHOD_LABEL)          METHOD_LABEL=${VALUE} ;;  # eg. vae-kl or bcl
            BASE_CONFIG)           BASE_CONFIG=${VALUE} ;;
            LANG_DIR)              LANG_DIR=${VALUE} ;;
            ASSUME_COMPLETE)       ASSUME_COMPLETE=${VALUE} ;;
            TRAIN_DATA)            TRAIN_DATA=${VALUE} ;;
            DEV_DATA)              DEV_DATA=${VALUE} ;;
            TEST_DATA)             TEST_DATA=${VALUE} ;;
            VOCAB_PATH)            VOCAB_PATH=${VALUE} ;;
            MODEL_NAME)            MODEL_NAME=${VALUE} ;; 
            PAD_TOKEN)             PAD_TOKEN=${VALUE} ;; 
            OOV_TOKEN)             OOV_TOKEN=${VALUE} ;;
            BATCH_SIZE)            BATCH_SIZE=${VALUE} ;;
            VALIDATION_BATCH_SIZE) VALIDATION_BATCH_SIZE=${VALUE} ;;  
            RANDOM_SEED)           RANDOM_SEED=${VALUE} ;; 
            DROPOUT)               DROPOUT=${VALUE} ;; 
            LR)                    LR=${VALUE} ;; 
            NUM_EPOCHS)            NUM_EPOCHS=${VALUE} ;; 
            PRIOR_TYPE)            PRIOR_TYPE=${VALUE} ;; 
            PRIOR_WEIGHT)          PRIOR_WEIGHT=${VALUE} ;;
            ENTITY_RATIO)          ENTITY_RATIO=${VALUE} ;; 
            ENTITY_RATIO_MARGIN)   ENTITY_RATIO_MARGIN=${VALUE} ;; 
            *)   
    esac    
done

# Navigate to dir
cd /home/ubuntu/ner-expected-entity-ratio
which python

# Setup
if $IS_REMOTE
then
  MAIN_IP="172.31.50.140"  # change this to your main server internal ip
  ssh-keygen -R $MAIN_IP
  ssh-keyscan $MAIN_IP >> ~/.ssh/known_hosts
fi
EXPERIMENT_DIR=experiments/$LANG_LABEL/$DATASET_LABEL
mkdir -p $EXPERIMENT_DIR
mkdir -p $LANG_DIR
TRAIN_DATA_PATH=$LANG_DIR/$TRAIN_DATA
DEV_DATA_PATH=$LANG_DIR/$DEV_DATA
TEST_DATA_PATH=$LANG_DIR/$TEST_DATA
SERIALIZATION_DIR=$EXPERIMENT_DIR/$METHOD_LABEL

# Some prints
echo GOT THE FOLLOWING PARAMS:
echo EXPERIMENT_DIR=$EXPERIMENT_DIR
echo SERIALIZATION_DIR=$SERIALIZATION_DIR
echo LANG_LABEL=$LANG_LABEL
echo DATASET_LABEL=$DATASET_LABEL
echo METHOD_LABEL=$METHOD_LABEL
echo BASE_CONFIG=$BASE_CONFIG
echo ASSUME_COMPLETE=$ASSUME_COMPLETE
echo LANG_DIR=$LANG_DIR
echo TRAIN_DATA=$TRAIN_DATA
echo DEV_DATA=$DEV_DATA
echo TEST_DATA=$TEST_DATA
echo VOCAB_PATH=$VOCAB_PATH
echo MODEL_NAME=$MODEL_NAME
echo PAD_TOKEN=$PAD_TOKEN
echo OOV_TOKEN=$OOV_TOKEN
echo BATCH_SIZE=$BATCH_SIZE
echo VALIDATION_BATCH_SIZE=$VALIDATION_BATCH_SIZE
echo RANDOM_SEED=$RANDOM_SEED
echo DROPOUT=$DROPOUT
echo LR=$LR
echo NUM_EPOCHS=$NUM_EPOCHS
echo PRIOR_TYPE=$PRIOR_TYPE
echo PRIOR_WEIGHT=$PRIOR_WEIGHT
echo ENTITY_RATIO=$ENTITY_RATIO
echo ENTITY_RATIO_MARGIN=$ENTITY_RATIO_MARGIN

# Some exports
export ASSUME_COMPLETE=$ASSUME_COMPLETE
export TRAIN_DATA_PATH=$TRAIN_DATA_PATH
export DEV_DATA_PATH=$DEV_DATA_PATH
export TEST_DATA_PATH=$TEST_DATA_PATH
export VOCAB_PATH=$VOCAB_PATH
export MODEL_NAME=$MODEL_NAME
export PAD_TOKEN=$PAD_TOKEN
export OOV_TOKEN=$OOV_TOKEN
export BATCH_SIZE=$BATCH_SIZE
export VALIDATION_BATCH_SIZE=$VALIDATION_BATCH_SIZE
export RANDOM_SEED=$RANDOM_SEED
export DROPOUT=$DROPOUT
export LR=$LR
export NUM_EPOCHS=$NUM_EPOCHS
export PRIOR_TYPE=$PRIOR_TYPE
export PRIOR_WEIGHT=$PRIOR_WEIGHT
export ENTITY_RATIO=$ENTITY_RATIO
export ENTITY_RATIO_MARGIN=$ENTITY_RATIO_MARGIN


# Download all needed data from main node
if $IS_REMOTE
then
  echo Downloading Data
  mkdir -p $VOCAB_PATH
  scp -i /home/ubuntu/aws-ec2-mcollins.pem ubuntu@$MAIN_IP:~/ner-expected-entity-ratio/$TRAIN_DATA_PATH $TRAIN_DATA_PATH
  scp -i /home/ubuntu/aws-ec2-mcollins.pem ubuntu@$MAIN_IP:~/ner-expected-entity-ratio/$DEV_DATA_PATH $DEV_DATA_PATH
  scp -i /home/ubuntu/aws-ec2-mcollins.pem ubuntu@$MAIN_IP:~/ner-expected-entity-ratio/$TEST_DATA_PATH $TEST_DATA_PATH
  scp -r -i /home/ubuntu/aws-ec2-mcollins.pem ubuntu@$MAIN_IP:~/ner-expected-entity-ratio/$VOCAB_PATH/* $VOCAB_PATH/
fi

# Run a jupyter server so we can check in on things, copying our password config
if $RUN_JUPYTER
then
  echo Running Jupyter
  scp -i /home/ubuntu/aws-ec2-mcollins.pem ubuntu@$MAIN_IP:/home/ubuntu/.jupyter/jupyter_notebook_config.json /home/ubuntu/.jupyter/jupyter_notebook_config.json
  /home/ubuntu/anaconda3/envs/env/bin/jupyter notebook --ip 0.0.0.0 --no-browser &> logs/jupyter.log &
fi

# Run a tensorboard instance watching the experiment dir
if $RUN_TENSORBOARD
then
  echo Running Tensorboard
  /home/ubuntu/anaconda3/envs/env/bin/tensorboard --host 0.0.0.0 --logdir $EXPERIMENT_DIR &> logs/tensorboard.log &
fi

# Run the experiment
echo Running Experiment, check at logs/train_${LANG_LABEL}_${DATASET_LABEL}_${METHOD_LABEL}.out
/home/ubuntu/anaconda3/envs/env/bin/allennlp train $BASE_CONFIG\
  -f -s $SERIALIZATION_DIR\
  --include ml\
  --file-friendly-logging\
  &> logs/train_${LANG_LABEL}_${DATASET_LABEL}_${METHOD_LABEL}.out

echo Experiment finished. 
echo Uploading train logs to logs/train_remote_${LANG_LABEL}_${DATASET_LABEL}_${METHOD_LABEL}.out
echo Uploading final results bach to $SERIALIZATION_DIR
if $IS_REMOTE
then
  # Then send the results back to the main node
  scp -i /home/ubuntu/aws-ec2-mcollins.pem logs/train_${LANG_LABEL}_${DATASET_LABEL}_${METHOD_LABEL}.out ubuntu@$MAIN_IP:~/ner-expected-entity-ratio/logs/train_remote_${LANG_LABEL}_${DATASET_LABEL}_${METHOD_LABEL}.out
  scp -r -i /home/ubuntu/aws-ec2-mcollins.pem $SERIALIZATION_DIR ubuntu@$MAIN_IP:~/ner-expected-entity-ratio/$SERIALIZATION_DIR
fi