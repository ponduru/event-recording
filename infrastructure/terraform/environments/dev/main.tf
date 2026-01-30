# Prismata Dev Environment

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
  }

  # Uncomment to use remote state
  # backend "s3" {
  #   bucket         = "prismata-terraform-state"
  #   key            = "dev/terraform.tfstate"
  #   region         = "us-east-1"
  #   dynamodb_table = "prismata-terraform-locks"
  #   encrypt        = true
  # }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "prismata"
}

variable "environment" {
  description = "Environment"
  type        = string
  default     = "dev"
}

variable "container_image" {
  description = "Streamlit container image"
  type        = string
  default     = ""  # Set after building Docker image
}

# VPC
module "vpc" {
  source = "../../modules/vpc"

  project_name       = var.project_name
  environment        = var.environment
  vpc_cidr           = "10.0.0.0/16"
  availability_zones = ["${var.aws_region}a", "${var.aws_region}b"]
}

# S3 Buckets
module "s3" {
  source = "../../modules/s3"

  project_name           = var.project_name
  environment            = var.environment
  enable_versioning      = true
  lifecycle_glacier_days = 0  # Disable Glacier for dev
}

# RDS PostgreSQL
module "rds" {
  source = "../../modules/rds"

  project_name               = var.project_name
  environment                = var.environment
  vpc_id                     = module.vpc.vpc_id
  subnet_ids                 = module.vpc.private_subnet_ids
  allowed_security_group_ids = [module.ecs.ecs_security_group_id, module.ec2_gpu.security_group_id]

  instance_class          = "db.t3.medium"
  allocated_storage       = 20
  multi_az                = false
  backup_retention_period = 7
  deletion_protection     = false
}

# ECS for Streamlit UI
module "ecs" {
  source = "../../modules/ecs"

  project_name       = var.project_name
  environment        = var.environment
  vpc_id             = module.vpc.vpc_id
  public_subnet_ids  = module.vpc.public_subnet_ids
  private_subnet_ids = module.vpc.private_subnet_ids

  container_image = var.container_image != "" ? var.container_image : "public.ecr.aws/docker/library/python:3.11-slim"
  container_port  = 8501
  cpu             = 2048
  memory          = 4096
  desired_count   = 1

  s3_bucket_arns = module.s3.all_bucket_arns
  db_secret_arn  = module.rds.db_secret_arn

  environment_variables = {
    PRISMATA_STORAGE_BACKEND   = "s3"
    PRISMATA_S3_BUCKET_VIDEOS     = module.s3.videos_bucket_name
    PRISMATA_S3_BUCKET_LABELS     = module.s3.labels_bucket_name
    PRISMATA_S3_BUCKET_MODELS     = module.s3.models_bucket_name
    PRISMATA_S3_BUCKET_DETECTIONS = module.s3.detections_bucket_name
    AWS_REGION                    = var.aws_region
  }

  secrets = {
    DATABASE_URL = "${module.rds.db_secret_arn}:database_url::"
  }
}

# EC2 GPU for Training/Inference
module "ec2_gpu" {
  source = "../../modules/ec2-gpu"

  project_name   = var.project_name
  environment    = var.environment
  vpc_id         = module.vpc.vpc_id
  subnet_id      = module.vpc.private_subnet_ids[0]
  instance_type  = "g4dn.xlarge"
  s3_bucket_arns = module.s3.all_bucket_arns
  db_secret_arn  = module.rds.db_secret_arn
}

# Lambda Orchestrator
module "lambda_orchestrator" {
  source = "../../modules/lambda-orchestrator"

  project_name           = var.project_name
  environment            = var.environment
  gpu_launch_template_id = module.ec2_gpu.launch_template_id
  gpu_subnet_id          = module.vpc.private_subnet_ids[0]
  gpu_security_group_id  = module.ec2_gpu.security_group_id
  s3_bucket_arns         = module.s3.all_bucket_arns
  auto_terminate_minutes = 30
}

# Outputs
output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "alb_dns_name" {
  description = "ALB DNS name for Streamlit UI"
  value       = module.ecs.alb_dns_name
}

output "database_endpoint" {
  description = "RDS endpoint"
  value       = module.rds.db_instance_endpoint
  sensitive   = true
}

output "s3_buckets" {
  description = "S3 bucket names"
  value = {
    videos     = module.s3.videos_bucket_name
    labels     = module.s3.labels_bucket_name
    models     = module.s3.models_bucket_name
    detections = module.s3.detections_bucket_name
  }
}

output "lambda_orchestrator_arn" {
  description = "Lambda orchestrator ARN"
  value       = module.lambda_orchestrator.function_arn
}

output "gpu_launch_template_id" {
  description = "GPU launch template ID"
  value       = module.ec2_gpu.launch_template_id
}

output "environment_variables" {
  description = "Environment variables for local development"
  value = {
    PRISMATA_STORAGE_BACKEND      = "s3"
    PRISMATA_S3_BUCKET_VIDEOS     = module.s3.videos_bucket_name
    PRISMATA_S3_BUCKET_LABELS     = module.s3.labels_bucket_name
    PRISMATA_S3_BUCKET_MODELS     = module.s3.models_bucket_name
    PRISMATA_S3_BUCKET_DETECTIONS = module.s3.detections_bucket_name
    PRISMATA_DB_HOST              = module.rds.db_instance_address
    PRISMATA_DB_PORT              = module.rds.db_instance_port
    PRISMATA_DB_NAME              = "prismata"
    AWS_REGION                    = var.aws_region
    PRISMATA_LAMBDA_ORCHESTRATOR_ARN = module.lambda_orchestrator.function_arn
    PRISMATA_GPU_LAUNCH_TEMPLATE  = module.ec2_gpu.launch_template_id
    PRISMATA_GPU_SUBNET_ID        = module.vpc.private_subnet_ids[0]
    PRISMATA_GPU_SECURITY_GROUP   = module.ec2_gpu.security_group_id
    PRISMATA_GPU_IAM_PROFILE      = module.ec2_gpu.instance_profile_name
    PRISMATA_GPU_AMI_ID           = module.ec2_gpu.ami_id
  }
}
