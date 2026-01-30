# EC2 GPU Module for Prismata Training/Inference

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

variable "vpc_id" {
  description = "VPC ID"
  type        = string
}

variable "subnet_id" {
  description = "Subnet ID for GPU instances"
  type        = string
}

variable "instance_type" {
  description = "GPU instance type"
  type        = string
  default     = "g4dn.xlarge"
}

variable "key_pair_name" {
  description = "SSH key pair name"
  type        = string
  default     = ""
}

variable "s3_bucket_arns" {
  description = "S3 bucket ARNs for IAM policy"
  type        = list(string)
  default     = []
}

variable "db_secret_arn" {
  description = "Database secret ARN"
  type        = string
  default     = ""
}

locals {
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

# Security Group for GPU instances
resource "aws_security_group" "gpu" {
  name        = "${var.project_name}-${var.environment}-gpu-sg"
  description = "Security group for GPU instances"
  vpc_id      = var.vpc_id

  # SSH access (optional, for debugging)
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]  # VPC only
  }

  # All outbound
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, {
    Name = "${var.project_name}-${var.environment}-gpu-sg"
  })
}

# IAM Role for GPU instances
resource "aws_iam_role" "gpu" {
  name = "${var.project_name}-${var.environment}-gpu-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = local.common_tags
}

# S3 access policy
resource "aws_iam_role_policy" "gpu_s3" {
  count = length(var.s3_bucket_arns) > 0 ? 1 : 0
  name  = "s3-access"
  role  = aws_iam_role.gpu.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = concat(
          var.s3_bucket_arns,
          [for arn in var.s3_bucket_arns : "${arn}/*"]
        )
      }
    ]
  })
}

# Secrets Manager access
resource "aws_iam_role_policy" "gpu_secrets" {
  count = var.db_secret_arn != "" ? 1 : 0
  name  = "secrets-access"
  role  = aws_iam_role.gpu.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = [var.db_secret_arn]
      }
    ]
  })
}

# CloudWatch Logs access
resource "aws_iam_role_policy" "gpu_logs" {
  name = "cloudwatch-logs"
  role = aws_iam_role.gpu.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Resource = "arn:aws:logs:*:*:log-group:/prismata/*"
      }
    ]
  })
}

# EC2 self-terminate permission
resource "aws_iam_role_policy" "gpu_ec2" {
  name = "ec2-self-terminate"
  role = aws_iam_role.gpu.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ec2:TerminateInstances"
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "ec2:ResourceTag/Project" = var.project_name
          }
        }
      },
      {
        Effect = "Allow"
        Action = [
          "ec2:DescribeInstances",
          "ec2:DescribeTags"
        ]
        Resource = "*"
      }
    ]
  })
}

# SSM access for remote command execution
resource "aws_iam_role_policy_attachment" "gpu_ssm" {
  role       = aws_iam_role.gpu.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

# Instance profile
resource "aws_iam_instance_profile" "gpu" {
  name = "${var.project_name}-${var.environment}-gpu-profile"
  role = aws_iam_role.gpu.name
}

# Get latest Deep Learning AMI
data "aws_ami" "deep_learning" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning AMI GPU PyTorch *-Ubuntu 22.04-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

# Launch Template for GPU instances
resource "aws_launch_template" "gpu" {
  name = "${var.project_name}-${var.environment}-gpu-template"

  image_id      = data.aws_ami.deep_learning.id
  instance_type = var.instance_type

  iam_instance_profile {
    name = aws_iam_instance_profile.gpu.name
  }

  network_interfaces {
    associate_public_ip_address = false
    security_groups             = [aws_security_group.gpu.id]
    subnet_id                   = var.subnet_id
  }

  dynamic "key_name" {
    for_each = var.key_pair_name != "" ? [1] : []
    content {
      key_name = var.key_pair_name
    }
  }

  block_device_mappings {
    device_name = "/dev/sda1"

    ebs {
      volume_size           = 100
      volume_type           = "gp3"
      delete_on_termination = true
      encrypted             = true
    }
  }

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"  # IMDSv2
    http_put_response_hop_limit = 1
  }

  monitoring {
    enabled = true
  }

  tag_specifications {
    resource_type = "instance"
    tags = merge(local.common_tags, {
      Name          = "${var.project_name}-${var.environment}-gpu"
      AutoTerminate = "true"
    })
  }

  tag_specifications {
    resource_type = "volume"
    tags = local.common_tags
  }

  tags = local.common_tags
}

# CloudWatch Log Group for GPU jobs
resource "aws_cloudwatch_log_group" "gpu_jobs" {
  name              = "/prismata/jobs"
  retention_in_days = 30

  tags = local.common_tags
}

# Outputs
output "launch_template_id" {
  description = "GPU launch template ID"
  value       = aws_launch_template.gpu.id
}

output "launch_template_version" {
  description = "GPU launch template latest version"
  value       = aws_launch_template.gpu.latest_version
}

output "security_group_id" {
  description = "GPU security group ID"
  value       = aws_security_group.gpu.id
}

output "instance_profile_name" {
  description = "GPU instance profile name"
  value       = aws_iam_instance_profile.gpu.name
}

output "instance_profile_arn" {
  description = "GPU instance profile ARN"
  value       = aws_iam_instance_profile.gpu.arn
}

output "ami_id" {
  description = "Deep Learning AMI ID"
  value       = data.aws_ami.deep_learning.id
}

output "log_group_name" {
  description = "CloudWatch log group for GPU jobs"
  value       = aws_cloudwatch_log_group.gpu_jobs.name
}
