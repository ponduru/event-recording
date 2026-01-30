# Lambda Orchestrator Module for GPU Instance Lifecycle

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

variable "project_name" {
  description = "Project name for tagging"
  type        = string
  default     = "prismata"
}

variable "environment" {
  description = "Environment (dev, prod)"
  type        = string
}

variable "gpu_launch_template_id" {
  description = "GPU instance launch template ID"
  type        = string
}

variable "gpu_subnet_id" {
  description = "Subnet ID for GPU instances"
  type        = string
}

variable "gpu_security_group_id" {
  description = "Security group ID for GPU instances"
  type        = string
}

variable "s3_bucket_arns" {
  description = "S3 bucket ARNs"
  type        = list(string)
  default     = []
}

variable "auto_terminate_minutes" {
  description = "Auto-terminate instances after this many minutes"
  type        = number
  default     = 30
}

locals {
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }

  function_name = "${var.project_name}-${var.environment}-gpu-orchestrator"
}

# IAM Role for Lambda
resource "aws_iam_role" "lambda" {
  name = "${var.project_name}-${var.environment}-gpu-orchestrator-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })

  tags = local.common_tags
}

# Lambda basic execution
resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# EC2 permissions for Lambda
resource "aws_iam_role_policy" "lambda_ec2" {
  name = "ec2-access"
  role = aws_iam_role.lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ec2:RunInstances",
          "ec2:TerminateInstances",
          "ec2:DescribeInstances",
          "ec2:DescribeInstanceStatus",
          "ec2:CreateTags"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "iam:PassRole"
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "iam:PassedToService" = "ec2.amazonaws.com"
          }
        }
      }
    ]
  })
}

# SSM permissions for Lambda
resource "aws_iam_role_policy" "lambda_ssm" {
  name = "ssm-access"
  role = aws_iam_role.lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ssm:SendCommand",
          "ssm:GetCommandInvocation"
        ]
        Resource = "*"
      }
    ]
  })
}

# Lambda function code
data "archive_file" "lambda" {
  type        = "zip"
  output_path = "${path.module}/lambda.zip"

  source {
    content  = <<-PYTHON
import json
import boto3
import os
import time
from datetime import datetime, timedelta

ec2 = boto3.client('ec2')
ssm = boto3.client('ssm')

LAUNCH_TEMPLATE_ID = os.environ['LAUNCH_TEMPLATE_ID']
SUBNET_ID = os.environ['SUBNET_ID']
PROJECT_NAME = os.environ['PROJECT_NAME']
ENVIRONMENT = os.environ['ENVIRONMENT']
AUTO_TERMINATE_MINUTES = int(os.environ.get('AUTO_TERMINATE_MINUTES', '30'))


def lambda_handler(event, context):
    """Handle GPU orchestration requests.

    Actions:
    - start_job: Launch a new GPU instance for a job
    - check_job: Check status of a running job
    - terminate_job: Terminate a job's instance
    - cleanup_stale: Terminate instances running too long
    """
    action = event.get('action', 'start_job')

    if action == 'start_job':
        return start_job(event)
    elif action == 'check_job':
        return check_job(event)
    elif action == 'terminate_job':
        return terminate_job(event)
    elif action == 'cleanup_stale':
        return cleanup_stale(event)
    else:
        return {'error': f'Unknown action: {action}'}


def start_job(event):
    """Start a new GPU instance for a job."""
    job_id = event.get('job_id', f"job-{int(time.time())}")
    job_type = event.get('job_type', 'training')
    job_config = event.get('config', {})

    # Create user data script
    user_data = create_user_data(job_id, job_type, job_config)

    # Launch instance
    response = ec2.run_instances(
        LaunchTemplate={
            'LaunchTemplateId': LAUNCH_TEMPLATE_ID,
            'Version': '$Latest'
        },
        MinCount=1,
        MaxCount=1,
        SubnetId=SUBNET_ID,
        UserData=user_data,
        TagSpecifications=[
            {
                'ResourceType': 'instance',
                'Tags': [
                    {'Key': 'Name', 'Value': f'{PROJECT_NAME}-gpu-{job_id[:8]}'},
                    {'Key': 'Project', 'Value': PROJECT_NAME},
                    {'Key': 'Environment', 'Value': ENVIRONMENT},
                    {'Key': 'JobId', 'Value': job_id},
                    {'Key': 'JobType', 'Value': job_type},
                    {'Key': 'AutoTerminate', 'Value': 'true'},
                    {'Key': 'LaunchTime', 'Value': datetime.utcnow().isoformat()},
                ]
            }
        ]
    )

    instance_id = response['Instances'][0]['InstanceId']

    return {
        'status': 'started',
        'job_id': job_id,
        'instance_id': instance_id
    }


def create_user_data(job_id, job_type, config):
    """Create user data script for the instance."""
    import base64

    config_json = json.dumps({
        'job_id': job_id,
        'job_type': job_type,
        **config
    })

    script = f'''#!/bin/bash
set -e

# Log to CloudWatch
exec > >(tee /var/log/prismata-job.log) 2>&1

echo "Starting Prismata GPU job: {job_id}"
echo "Job type: {job_type}"

# Write job config
cat > /tmp/job_config.json << 'CONFIGEOF'
{config_json}
CONFIGEOF

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate pytorch || true

# Install/update prismata
pip install git+https://github.com/YOUR_REPO/prismata.git || true

# Run the job
python -m src.aws.job_runner /tmp/job_config.json

# Signal completion and terminate
echo "Job completed, terminating instance"
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region {os.environ.get('AWS_REGION', 'us-east-1')}
'''

    return base64.b64encode(script.encode()).decode()


def check_job(event):
    """Check status of a job."""
    job_id = event.get('job_id')
    instance_id = event.get('instance_id')

    if instance_id:
        response = ec2.describe_instances(InstanceIds=[instance_id])
    else:
        response = ec2.describe_instances(
            Filters=[
                {'Name': 'tag:JobId', 'Values': [job_id]},
                {'Name': 'tag:Project', 'Values': [PROJECT_NAME]}
            ]
        )

    if not response['Reservations']:
        return {'status': 'not_found', 'job_id': job_id}

    instance = response['Reservations'][0]['Instances'][0]
    state = instance['State']['Name']

    return {
        'status': state,
        'job_id': job_id,
        'instance_id': instance['InstanceId'],
        'launch_time': instance.get('LaunchTime', '').isoformat() if instance.get('LaunchTime') else None
    }


def terminate_job(event):
    """Terminate a job's instance."""
    instance_id = event.get('instance_id')

    if not instance_id:
        return {'error': 'instance_id required'}

    ec2.terminate_instances(InstanceIds=[instance_id])

    return {
        'status': 'terminated',
        'instance_id': instance_id
    }


def cleanup_stale(event):
    """Terminate instances that have been running too long."""
    max_age_minutes = event.get('max_age_minutes', AUTO_TERMINATE_MINUTES)

    response = ec2.describe_instances(
        Filters=[
            {'Name': 'tag:Project', 'Values': [PROJECT_NAME]},
            {'Name': 'tag:AutoTerminate', 'Values': ['true']},
            {'Name': 'instance-state-name', 'Values': ['pending', 'running']}
        ]
    )

    terminated = []
    cutoff = datetime.utcnow() - timedelta(minutes=max_age_minutes)

    for reservation in response.get('Reservations', []):
        for instance in reservation.get('Instances', []):
            launch_time = instance.get('LaunchTime')
            if launch_time and launch_time.replace(tzinfo=None) < cutoff:
                instance_id = instance['InstanceId']
                ec2.terminate_instances(InstanceIds=[instance_id])
                terminated.append(instance_id)

    return {
        'status': 'cleanup_complete',
        'terminated': terminated,
        'count': len(terminated)
    }
PYTHON
    filename = "lambda_function.py"
  }
}

# Lambda Function
resource "aws_lambda_function" "orchestrator" {
  filename         = data.archive_file.lambda.output_path
  source_code_hash = data.archive_file.lambda.output_base64sha256
  function_name    = local.function_name
  role             = aws_iam_role.lambda.arn
  handler          = "lambda_function.lambda_handler"
  runtime          = "python3.11"
  timeout          = 300
  memory_size      = 256

  environment {
    variables = {
      LAUNCH_TEMPLATE_ID     = var.gpu_launch_template_id
      SUBNET_ID              = var.gpu_subnet_id
      PROJECT_NAME           = var.project_name
      ENVIRONMENT            = var.environment
      AUTO_TERMINATE_MINUTES = tostring(var.auto_terminate_minutes)
    }
  }

  tags = local.common_tags
}

# CloudWatch Event Rule for stale instance cleanup
resource "aws_cloudwatch_event_rule" "cleanup" {
  name                = "${var.project_name}-${var.environment}-gpu-cleanup"
  description         = "Cleanup stale GPU instances"
  schedule_expression = "rate(15 minutes)"

  tags = local.common_tags
}

resource "aws_cloudwatch_event_target" "cleanup" {
  rule      = aws_cloudwatch_event_rule.cleanup.name
  target_id = "cleanup-stale-instances"
  arn       = aws_lambda_function.orchestrator.arn

  input = jsonencode({
    action = "cleanup_stale"
  })
}

resource "aws_lambda_permission" "cleanup" {
  statement_id  = "AllowCloudWatchInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.orchestrator.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.cleanup.arn
}

# Outputs
output "function_arn" {
  description = "Lambda function ARN"
  value       = aws_lambda_function.orchestrator.arn
}

output "function_name" {
  description = "Lambda function name"
  value       = aws_lambda_function.orchestrator.function_name
}

output "invoke_arn" {
  description = "Lambda invoke ARN"
  value       = aws_lambda_function.orchestrator.invoke_arn
}
