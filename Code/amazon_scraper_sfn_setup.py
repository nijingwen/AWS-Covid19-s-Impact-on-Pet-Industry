import boto3
import json

def make_def(lambda_arn):
    definition = {
      "Comment": "Amazon Scraper State Machine",
      "StartAt": "Map",
      "States": {
        "Map": {
          "Type": "Map",
          "End": True,
          "Iterator": {
            "StartAt": "Lambda Invoke",
            "States": {
              "Lambda Invoke": {
                "Type": "Task",
                "Resource": "arn:aws:states:::lambda:invoke",
                "OutputPath": "$.Payload",
                "Parameters": {
                  "Payload.$": "$",
                  "FunctionName": lambda_arn
                },
                "Retry": [
                  {
                    "ErrorEquals": [
                      "Lambda.ServiceException",
                      "Lambda.AWSLambdaException",
                      "Lambda.SdkClientException",
                      "Lambda.TooManyRequestsException",
                      "States.TaskFailed"
                    ],
                    "IntervalSeconds": 2,
                    "MaxAttempts": 6,
                    "BackoffRate": 2
                  }
                ],
                "End": True
              }
            }
          }
        }
      }
    }
    return definition

if __name__ == '__main__':
    iam = boto3.client('iam')
    sfn = boto3.client('stepfunctions')
    aws_lambda = boto3.client('lambda')

    lambda_arn = [f['FunctionArn']
                  for f in aws_lambda.list_functions()['Functions']
                  if f['FunctionName'] == 'amazon_scraper'][0]
    role = iam.get_role(RoleName='LabRole')

    # Use Lambda ARN to create State Machine Definition
    sf_def = make_def(lambda_arn)

    # Create Step Function State Machine and call it 'q2'
    response = sfn.create_state_machine(
            name='amazon_scraper',
            definition=json.dumps(sf_def),
            roleArn=role['Role']['Arn'],
            type='EXPRESS'
        )
