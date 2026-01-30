# S3 Module for Prismata
# Creates buckets for videos, labels, models, and detections

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

variable "enable_versioning" {
  description = "Enable versioning on buckets"
  type        = bool
  default     = true
}

variable "lifecycle_glacier_days" {
  description = "Days before transitioning to Glacier (0 to disable)"
  type        = number
  default     = 90
}

locals {
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }

  bucket_names = {
    videos     = "${var.project_name}-videos-${var.environment}"
    labels     = "${var.project_name}-labels-${var.environment}"
    models     = "${var.project_name}-models-${var.environment}"
    detections = "${var.project_name}-detections-${var.environment}"
  }
}

# Videos Bucket
resource "aws_s3_bucket" "videos" {
  bucket = local.bucket_names.videos

  tags = merge(local.common_tags, {
    Name = local.bucket_names.videos
    Type = "videos"
  })
}

resource "aws_s3_bucket_versioning" "videos" {
  bucket = aws_s3_bucket.videos.id
  versioning_configuration {
    status = var.enable_versioning ? "Enabled" : "Disabled"
  }
}

resource "aws_s3_bucket_intelligent_tiering_configuration" "videos" {
  bucket = aws_s3_bucket.videos.id
  name   = "EntireBucket"

  tiering {
    access_tier = "DEEP_ARCHIVE_ACCESS"
    days        = 180
  }
  tiering {
    access_tier = "ARCHIVE_ACCESS"
    days        = 90
  }
}

resource "aws_s3_bucket_cors_configuration" "videos" {
  bucket = aws_s3_bucket.videos.id

  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["GET", "HEAD"]
    allowed_origins = ["*"]
    expose_headers  = ["ETag", "Content-Length"]
    max_age_seconds = 3600
  }
}

# Labels Bucket
resource "aws_s3_bucket" "labels" {
  bucket = local.bucket_names.labels

  tags = merge(local.common_tags, {
    Name = local.bucket_names.labels
    Type = "labels"
  })
}

resource "aws_s3_bucket_versioning" "labels" {
  bucket = aws_s3_bucket.labels.id
  versioning_configuration {
    status = "Enabled"  # Always version labels
  }
}

# Models Bucket
resource "aws_s3_bucket" "models" {
  bucket = local.bucket_names.models

  tags = merge(local.common_tags, {
    Name = local.bucket_names.models
    Type = "models"
  })
}

resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id
  versioning_configuration {
    status = "Enabled"  # Always version models
  }
}

# Detections Bucket
resource "aws_s3_bucket" "detections" {
  bucket = local.bucket_names.detections

  tags = merge(local.common_tags, {
    Name = local.bucket_names.detections
    Type = "detections"
  })
}

resource "aws_s3_bucket_versioning" "detections" {
  bucket = aws_s3_bucket.detections.id
  versioning_configuration {
    status = var.enable_versioning ? "Enabled" : "Disabled"
  }
}

# Lifecycle rules for detections (clean up old results)
resource "aws_s3_bucket_lifecycle_configuration" "detections" {
  bucket = aws_s3_bucket.detections.id

  rule {
    id     = "expire-old-detections"
    status = "Enabled"

    filter {
      prefix = ""
    }

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    dynamic "transition" {
      for_each = var.lifecycle_glacier_days > 0 ? [1] : []
      content {
        days          = var.lifecycle_glacier_days
        storage_class = "GLACIER"
      }
    }
  }
}

# Block public access on all buckets
resource "aws_s3_bucket_public_access_block" "videos" {
  bucket = aws_s3_bucket.videos.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "labels" {
  bucket = aws_s3_bucket.labels.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "models" {
  bucket = aws_s3_bucket.models.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "detections" {
  bucket = aws_s3_bucket.detections.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Server-side encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "videos" {
  bucket = aws_s3_bucket.videos.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "labels" {
  bucket = aws_s3_bucket.labels.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "models" {
  bucket = aws_s3_bucket.models.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "detections" {
  bucket = aws_s3_bucket.detections.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Outputs
output "videos_bucket_name" {
  description = "Videos bucket name"
  value       = aws_s3_bucket.videos.id
}

output "videos_bucket_arn" {
  description = "Videos bucket ARN"
  value       = aws_s3_bucket.videos.arn
}

output "labels_bucket_name" {
  description = "Labels bucket name"
  value       = aws_s3_bucket.labels.id
}

output "labels_bucket_arn" {
  description = "Labels bucket ARN"
  value       = aws_s3_bucket.labels.arn
}

output "models_bucket_name" {
  description = "Models bucket name"
  value       = aws_s3_bucket.models.id
}

output "models_bucket_arn" {
  description = "Models bucket ARN"
  value       = aws_s3_bucket.models.arn
}

output "detections_bucket_name" {
  description = "Detections bucket name"
  value       = aws_s3_bucket.detections.id
}

output "detections_bucket_arn" {
  description = "Detections bucket ARN"
  value       = aws_s3_bucket.detections.arn
}

output "all_bucket_arns" {
  description = "All bucket ARNs for IAM policies"
  value = [
    aws_s3_bucket.videos.arn,
    aws_s3_bucket.labels.arn,
    aws_s3_bucket.models.arn,
    aws_s3_bucket.detections.arn,
  ]
}
