# End to End implementation with MLOPS concept
# Electoral_Symbols_And_Vote_Detection

## Demo
1. Download the single any image from the artifacts/test sets
2. Upload that image to the application and submit for the prediction

## note 
    Currently we have implemented only FASTER RCNN model from pytorch
    URL: https://pytorch.org/vision/master/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html#torchvision.models.detection.fasterrcnn_resnet50_fpn_v2


## Workflows followed

1. create config.yaml
2. create secrets.yaml [Optional] not created.
3. create params.yaml
4. create enity
5. create configuration manager in src config
6. create components
7. create the pipeline
8. create main.py
9. create dvc.yaml


## dagshub setup

````bash
os.environ["MLFLOW_TRACKING_URI"]= "https://dagshub.com/surajacharya62/Electoral_Symbols_And_Vote_Detection.mlflow"
os.environ["MLFLOW_TRACKING_USRNAME"] = "surajacharya62"
os.environ["MLFLOW_TRACKING_PASSWORD"]="bae467a5cfadb6e0dee59eb7356f7ad1cc5abaf0"

# export MLFLOW_TRACKING_URI=https://dagshub.com/surajacharya62/Electoral_Symbols_And_Vote_Detection.mlflow
# export MLFLOW_TRACKING_USRNAME=surajacharya62
# export MLFLOW_TRACKING_PASSWORD=bae467a5cfadb6e0dee59eb7356f7ad1cc5abaf0

````





## AWS-CICD-Deployment-with-Github-Actions
1. Login to AWS console.
2. Create IAM user for deployment
    ## Policy required for accessing ECR and EC2 to IAM user:
    1. AmazonEC2ContainerRegistryFullAccess
    2. AmazonEC2FullAccess
3. create ECR(Elastic Container registry to save your docker image in aws)
    - Save the URI: 566373416292.dkr.ecr.us-east-1.amazonaws.com/ecr_ballot
3. Launch EC2(Ubuntu)
    ## Setting up the docker in EC2
    
    ## optional
    1. sudo apt-get update -y
    2. sudo apt-get upgrade

    ## required
    3. curl -fsSL https://get.docker.com -o get-docker.sh
    4. sudo sh get-docker.sh
    5. sudo usermod -aG docker ubuntu
    6. newgrp docker

## Configure EC2 as self-hosted runner in GitHubActions:
1. goto project setting-> 
2. goto actions > 
3. select runner> 
4. create new self hosted runner 
5. choose os
6. then run command one by one given there
7. Now goto "secrets and variables"
8. select actions
9. create new repository and provide the variables below
    ## Setting the user access key in github secrets in GitHubActions:
    AWS_ACCESS_KEY_ID =

    AWS_SECRET_ACCESS_KEY =

    AWS_REGION = us-east-1

    AWS_ECR_LOGIN_URI = ecr_ballot uri:640752516842.dkr.ecr.eu-north-1.amazonaws.com

    ECR_REPOSITORY_NAME = ecr_ballot


## Description: About the deployment

1. Build docker image of the source code

2. Push docker image to ECR

4. Pull image from ECR in EC2

5. Lauch docker image in EC2









