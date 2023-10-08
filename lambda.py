import subprocess

# pip install custom package to /tmp/ and add to path
subprocess.call('pip install --no-build-isolation --force-reinstall \
    "boto3>=1.28.57" \
    "awscli>=1.29.57" \
    "botocore>=1.31.57" -t /tmp/ --no-cache-dir'.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.call('pip install --quiet langchain==0.0.309 -t /tmp/ --no-cache-dir'.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
sys.path.insert(1, '/tmp/')

import json
import os
import sys
from langchain.llms.bedrock import Bedrock

import boto3

module_path = ".."
sys.path.append(os.path.abspath(module_path))
from utils import bedrock, print_ww


boto3_bedrock = bedrock.get_bedrock_client(
    assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
    region=os.environ.get("AWS_DEFAULT_REGION", None)
)

def lambda_handler(event, context):
    boto3_bedrock = bedrock.get_bedrock_client(
        assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
        region=os.environ.get("AWS_DEFAULT_REGION", None)
    )
    
    inference_modifier = {
        "max_tokens_to_sample": 4096,
        "temperature": 0.5,
        "top_k": 250,
        "top_p": 1,
        "stop_sequences": ["\n\nHuman"],
    }

    textgen_llm = Bedrock(
        model_id="anthropic.claude-v2",
        client=boto3_bedrock,
        model_kwargs=inference_modifier,
    )
    
    response = textgen_llm(event["Input"])
    return {"response" : response}
