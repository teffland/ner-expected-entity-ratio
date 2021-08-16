# Run run_ml_experiment.sh on a remote server specified by IP
# by doing this in a separate function and invoking the script on the remote server
# we can make sure that the experiment runs in nohup mode and doesn't depend on the login ssh connection

# NOTE: YOU MUST SPECIFY "$GITUSER" and "$GITPASS" IN A PARENT ENVIRONMENT

# Unpack key=value arguments
for ARGUMENT in "$@"
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)   
    case "$KEY" in
            PUBLIC_IP)             PUBLIC_IP=${VALUE} ;;
            PRIVATE_IP)            PRIVATE_IP=${VALUE} ;;
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



# Setup experiment and make sure we have a clean remote connection
EXPERIMENT_LABEL=${LANG_LABEL}_${DATASET_LABEL}_${METHOD_LABEL}
echo 
echo =====================================
echo Running ${EXPERIMENT_LABEL} at ${PUBLIC_IP}
echo check on run at: http://${PUBLIC_IP}:8888/terminals/1
echo check on run at: http://${PUBLIC_IP}:6006
echo local log at: logs/run_remote_${EXPERIMENT_LABEL}.out
mkdir -p experiments/$LANG_LABEL/$DATASET_LABEL
ssh-keygen -R $PRIVATE_IP &> /dev/null
ssh-keyscan $PRIVATE_IP &> /dev/null >> ~/.ssh/known_hosts
echo 

# Print out the run command
echo RUNNING COMMAND WITH PARAMS:
cat << EOF
bash scripts/run_ml_experiment.sh\
 IS_REMOTE=true\
 RUN_JUPYTER=$RUN_JUPYTER\
 RUN_TENSORBOARD=$RUN_TENSORBOARD\
 LANG_LABEL=$LANG_LABEL\
 DATASET_LABEL=$DATASET_LABEL\
 METHOD_LABEL=$METHOD_LABEL\
 BASE_CONFIG=$BASE_CONFIG\
 LANG_DIR=$LANG_DIR\
 ASSUME_COMPLETE=$ASSUME_COMPLETE\
 TRAIN_DATA=$TRAIN_DATA\
 DEV_DATA=$DEV_DATA\
 TEST_DATA=$TEST_DATA\
 VOCAB_PATH=$VOCAB_PATH\
 MODEL_NAME=$MODEL_NAME\
 PAD_TOKEN="$PAD_TOKEN"\
 OOV_TOKEN="$OOV_TOKEN"\
 BATCH_SIZE=$BATCH_SIZE\
 VALIDATION_BATCH_SIZE=$VALIDATION_BATCH_SIZE\
 RANDOM_SEED=$RANDOM_SEED\
 DROPOUT=$DROPOUT\
 LR=$LR\
 NUM_EPOCHS=$NUM_EPOCHS\
 PRIOR_TYPE=$PRIOR_TYPE\
 PRIOR_WEIGHT=$PRIOR_WEIGHT\
 ENTITY_RATIO=$ENTITY_RATIO\
 ENTITY_RATIO_MARGIN=$ENTITY_RATIO_MARGIN\
 &> logs/run_${EXPERIMENT_LABEL}.out &
EOF

# Login to the remove node and actually run the experiment
ssh -q -i /home/ubuntu/aws-ec2-mcollins.pem ubuntu@$PRIVATE_IP &> logs/run_remote_${EXPERIMENT_LABEL}.out << EOF
cd /home/ubuntu/ner-expected-entity-ratio
rm experiments/*jsonnet
git stash
rm dataset_stats.ipynb
rm dev_ontonotes_preprocessor.ipynb
git pull https://${GITUSER}:${GITPASS}@github.com/${GITUSER}/ner-expected-entity-ratio
ls experiments
nohup bash scripts/run_ml_experiment.sh\
 IS_REMOTE=true\
 RUN_JUPYTER=$RUN_JUPYTER\
 RUN_TENSORBOARD=$RUN_TENSORBOARD\
 LANG_LABEL=$LANG_LABEL\
 DATASET_LABEL=$DATASET_LABEL\
 METHOD_LABEL=$METHOD_LABEL\
 BASE_CONFIG=$BASE_CONFIG\
 LANG_DIR=$LANG_DIR\
 ASSUME_COMPLETE=$ASSUME_COMPLETE\
 TRAIN_DATA=$TRAIN_DATA\
 DEV_DATA=$DEV_DATA\
 TEST_DATA=$TEST_DATA\
 VOCAB_PATH=$VOCAB_PATH\
 MODEL_NAME=$MODEL_NAME\
 PAD_TOKEN="$PAD_TOKEN"\
 OOV_TOKEN="$OOV_TOKEN"\
 BATCH_SIZE=$BATCH_SIZE\
 VALIDATION_BATCH_SIZE=$VALIDATION_BATCH_SIZE\
 RANDOM_SEED=$RANDOM_SEED\
 DROPOUT=$DROPOUT\
 LR=$LR\
 NUM_EPOCHS=$NUM_EPOCHS\
 PRIOR_TYPE=$PRIOR_TYPE\
 PRIOR_WEIGHT=$PRIOR_WEIGHT\
 ENTITY_RATIO=$ENTITY_RATIO\
 ENTITY_RATIO_MARGIN=$ENTITY_RATIO_MARGIN\
 &> logs/run_${EXPERIMENT_LABEL}.out &
EOF
echo ===================================== 
