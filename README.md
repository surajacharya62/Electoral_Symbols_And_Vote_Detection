# Electoral_Symbols_And_Vote_Detection
End to End implementation with MLOPS concept

# Workflows followed

1. create config.yaml
2. create secrets.yaml [Optional] not created.
3. create params.yaml
4. create enity
5. create configuration manager in src config
6. create components
7. create the pipeline
8. create main.py
9. create dvc.yaml


### dagshub setup

````bash
os.environ["MLFLOW_TRACKING_URI"]= "https://dagshub.com/surajacharya62/Electoral_Symbols_And_Vote_Detection.mlflow"
os.environ["MLFLOW_TRACKING_USRNAME"] = "surajacharya62"
os.environ["MLFLOW_TRACKING_PASSWORD"]="bae467a5cfadb6e0dee59eb7356f7ad1cc5abaf0"

# export MLFLOW_TRACKING_URI=https://dagshub.com/surajacharya62/Electoral_Symbols_And_Vote_Detection.mlflow
# export MLFLOW_TRACKING_USRNAME=surajacharya62
# export MLFLOW_TRACKING_PASSWORD=bae467a5cfadb6e0dee59eb7356f7ad1cc5abaf0

````


ecr_ballot uri:640752516842.dkr.ecr.us-east-1.amazonaws.com/erc_ballot
