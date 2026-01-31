# Prismata Minimal Environment - Under $50/month
#
# Cost breakdown:
# - S3 (50GB): ~$2/month
# - RDS db.t3.micro: ~$15/month (or free tier first year)
# - EC2 t3.small for UI: ~$15/month (2GB RAM for inference)
# - GPU on-demand (2h): ~$4/month
# - No NAT Gateway (use public subnets + VPC endpoints)
# - No ALB (access EC2 directly)
# Total: ~$15-30/month

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
  default     = "minimal"
}

variable "ssh_key_name" {
  description = "SSH key pair name for EC2 access"
  type        = string
  default     = ""
}

variable "allowed_ip" {
  description = "Your IP address for SSH/HTTP access (use /32 CIDR)"
  type        = string
  default     = "0.0.0.0/0"  # Restrict this to your IP!
}

locals {
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

# =============================================================================
# VPC - Simple public-only setup (no NAT Gateway = saves ~$32/month)
# =============================================================================

resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = merge(local.common_tags, { Name = "${var.project_name}-vpc" })
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  tags   = merge(local.common_tags, { Name = "${var.project_name}-igw" })
}

resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = "${var.aws_region}a"
  map_public_ip_on_launch = true

  tags = merge(local.common_tags, { Name = "${var.project_name}-public" })
}

resource "aws_subnet" "public_b" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.2.0/24"
  availability_zone       = "${var.aws_region}b"
  map_public_ip_on_launch = true

  tags = merge(local.common_tags, { Name = "${var.project_name}-public-b" })
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = merge(local.common_tags, { Name = "${var.project_name}-public-rt" })
}

resource "aws_route_table_association" "public" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "public_b" {
  subnet_id      = aws_subnet.public_b.id
  route_table_id = aws_route_table.public.id
}

# VPC Endpoint for S3 (free, avoids data transfer costs)
resource "aws_vpc_endpoint" "s3" {
  vpc_id       = aws_vpc.main.id
  service_name = "com.amazonaws.${var.aws_region}.s3"

  route_table_ids = [aws_route_table.public.id]

  tags = merge(local.common_tags, { Name = "${var.project_name}-s3-endpoint" })
}

# =============================================================================
# S3 Buckets - Single bucket to minimize costs
# =============================================================================

resource "aws_s3_bucket" "data" {
  bucket = "${var.project_name}-data-${var.environment}-${random_id.bucket_suffix.hex}"

  tags = merge(local.common_tags, { Name = "${var.project_name}-data" })
}

resource "random_id" "bucket_suffix" {
  byte_length = 4
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket = aws_s3_bucket.data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Lifecycle rule to move old data to cheaper storage
resource "aws_s3_bucket_lifecycle_configuration" "data" {
  bucket = aws_s3_bucket.data.id

  rule {
    id     = "move-to-ia"
    status = "Enabled"

    filter {
      prefix = "videos/"
    }

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }
  }
}

# =============================================================================
# RDS PostgreSQL - db.t3.micro (~$15/month, free tier eligible)
# =============================================================================

resource "random_password" "db_password" {
  length  = 24
  special = false  # Simpler password for easier connection strings
}

resource "aws_db_subnet_group" "main" {
  name       = "${var.project_name}-db-subnet"
  subnet_ids = [aws_subnet.public.id, aws_subnet.public_b.id]

  tags = local.common_tags
}

resource "aws_security_group" "rds" {
  name        = "${var.project_name}-rds-sg"
  description = "RDS security group"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.app.id]
  }

  # Allow from your IP for direct access (optional)
  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ip]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, { Name = "${var.project_name}-rds-sg" })
}

resource "aws_db_instance" "main" {
  identifier = "${var.project_name}-${var.environment}"

  engine         = "postgres"
  engine_version = "15"
  instance_class = "db.t3.micro"  # Free tier eligible!

  allocated_storage     = 20
  max_allocated_storage = 50  # Auto-scale if needed
  storage_type          = "gp2"
  storage_encrypted     = true

  db_name  = "prismata"
  username = "prismata"
  password = random_password.db_password.result

  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.rds.id]
  publicly_accessible    = true  # Needed since no NAT Gateway

  skip_final_snapshot = true  # For dev/minimal environments
  deletion_protection = false

  backup_retention_period = 1
  backup_window          = "03:00-04:00"

  # Disable performance insights to save costs
  performance_insights_enabled = false

  tags = merge(local.common_tags, { Name = "${var.project_name}-db" })
}

# Store password in SSM Parameter Store (free, unlike Secrets Manager)
resource "aws_ssm_parameter" "db_password" {
  name  = "/${var.project_name}/${var.environment}/db-password"
  type  = "SecureString"
  value = random_password.db_password.result

  tags = local.common_tags
}

# =============================================================================
# EC2 for Streamlit UI - t3.micro (~$8/month, free tier eligible)
# =============================================================================

resource "aws_security_group" "app" {
  name        = "${var.project_name}-app-sg"
  description = "App security group"
  vpc_id      = aws_vpc.main.id

  # SSH - locked to your IP only
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ip]
  }

  # No inbound port 8501 — Streamlit is exposed via Cloudflare Tunnel only.
  # cloudflared makes an outbound connection, so only egress is needed.

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, { Name = "${var.project_name}-app-sg" })
}

# IAM Role for EC2
resource "aws_iam_role" "app" {
  name = "${var.project_name}-${var.environment}-app-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy" "app_s3" {
  name = "s3-access"
  role = aws_iam_role.app.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"]
      Resource = [aws_s3_bucket.data.arn, "${aws_s3_bucket.data.arn}/*"]
    }]
  })
}

resource "aws_iam_role_policy" "app_ssm" {
  name = "ssm-access"
  role = aws_iam_role.app.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["ssm:GetParameter", "ssm:GetParameters"]
      Resource = aws_ssm_parameter.db_password.arn
    }]
  })
}

resource "aws_iam_instance_profile" "app" {
  name = "${var.project_name}-${var.environment}-app-profile"
  role = aws_iam_role.app.name
}

# Get latest Amazon Linux 2023 AMI
data "aws_ami" "amazon_linux" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-2023*-x86_64"]
  }
}

resource "aws_instance" "app" {
  ami                    = data.aws_ami.amazon_linux.id
  instance_type          = "t3.small"  # ~$15/month, 2GB RAM needed for inference
  subnet_id              = aws_subnet.public.id
  vpc_security_group_ids = [aws_security_group.app.id]
  iam_instance_profile   = aws_iam_instance_profile.app.name
  key_name               = var.ssh_key_name != "" ? var.ssh_key_name : null

  root_block_device {
    volume_size = 20
    volume_type = "gp3"
  }

  user_data = base64encode(<<-EOF
    #!/bin/bash
    set -e

    # Install Docker
    yum update -y
    yum install -y docker git
    systemctl start docker
    systemctl enable docker
    usermod -aG docker ec2-user

    # Install Docker Compose
    curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose

    # Install cloudflared
    curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared
    chmod +x /usr/local/bin/cloudflared

    # Create systemd service for cloudflared tunnel
    cat > /etc/systemd/system/cloudflared-tunnel.service << 'UNIT'
    [Unit]
    Description=Cloudflare Tunnel to Streamlit
    After=network-online.target docker.service
    Wants=network-online.target

    [Service]
    Type=simple
    ExecStart=/usr/local/bin/cloudflared tunnel --url http://localhost:8501 --no-autoupdate
    Restart=on-failure
    RestartSec=10
    # Log the tunnel URL (contains the *.trycloudflare.com address)
    StandardOutput=journal
    StandardError=journal

    [Install]
    WantedBy=multi-user.target
    UNIT

    systemctl daemon-reload
    systemctl enable cloudflared-tunnel

    # Clone repo and deploy
    cd /home/ec2-user
    git clone https://github.com/ponduru/event-recording.git prismata
    cd prismata

    # Write .env file with Terraform-injected values
    cat > .env << 'ENVFILE'
    PRISMATA_STORAGE_BACKEND=s3
    PRISMATA_S3_BUCKET=${aws_s3_bucket.data.id}
    PRISMATA_DB_HOST=${aws_db_instance.main.address}
    PRISMATA_DB_PORT=5432
    PRISMATA_DB_NAME=prismata
    PRISMATA_DB_USER=prismata
    PRISMATA_DB_PASSWORD=${random_password.db_password.result}
    PRISMATA_DB_SSL_MODE=require
    AWS_REGION=${var.aws_region}
    ENVFILE

    # Fix indentation in .env (strip leading whitespace)
    sed -i 's/^[[:space:]]*//' .env

    # Build and run the Docker container
    docker build -t prismata .
    docker run -d \
      --name prismata \
      --restart unless-stopped \
      --env-file .env \
      -p 8501:8501 \
      prismata

    # Fix ownership
    chown -R ec2-user:ec2-user /home/ec2-user/prismata

    # Start the tunnel (it will wait for Streamlit to come up)
    systemctl start cloudflared-tunnel
  EOF
  )

  tags = merge(local.common_tags, { Name = "${var.project_name}-app" })
}

# Elastic IP for consistent access (free when attached to running instance)
resource "aws_eip" "app" {
  instance = aws_instance.app.id
  domain   = "vpc"

  tags = merge(local.common_tags, { Name = "${var.project_name}-app-eip" })
}

# =============================================================================
# GPU Instance Configuration (on-demand only, ~$0.53/hour)
# =============================================================================

resource "aws_security_group" "gpu" {
  name        = "${var.project_name}-gpu-sg"
  description = "GPU instance security group"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ip]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, { Name = "${var.project_name}-gpu-sg" })
}

resource "aws_iam_role" "gpu" {
  name = "${var.project_name}-${var.environment}-gpu-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy" "gpu_s3" {
  name = "s3-access"
  role = aws_iam_role.gpu.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
      Resource = [aws_s3_bucket.data.arn, "${aws_s3_bucket.data.arn}/*"]
    }]
  })
}

resource "aws_iam_role_policy" "gpu_ec2" {
  name = "ec2-self-terminate"
  role = aws_iam_role.gpu.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["ec2:TerminateInstances"]
        Resource = "*"
        Condition = {
          StringEquals = { "ec2:ResourceTag/Project" = var.project_name }
        }
      },
      {
        Effect   = "Allow"
        Action   = ["ec2:DescribeInstances"]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_instance_profile" "gpu" {
  name = "${var.project_name}-${var.environment}-gpu-profile"
  role = aws_iam_role.gpu.name
}

# Get Deep Learning AMI
data "aws_ami" "deep_learning" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning OSS Nvidia Driver AMI GPU PyTorch * (Ubuntu 22.04) *"]
  }
}

# GPU Launch Template (instances launched manually or via script)
resource "aws_launch_template" "gpu" {
  name = "${var.project_name}-gpu-template"

  image_id      = data.aws_ami.deep_learning.id
  instance_type = "g4dn.xlarge"  # $0.526/hour

  iam_instance_profile {
    name = aws_iam_instance_profile.gpu.name
  }

  network_interfaces {
    associate_public_ip_address = true
    security_groups             = [aws_security_group.gpu.id]
    subnet_id                   = aws_subnet.public.id
  }

  key_name = var.ssh_key_name != "" ? var.ssh_key_name : null

  block_device_mappings {
    device_name = "/dev/sda1"
    ebs {
      volume_size           = 100
      volume_type           = "gp3"
      delete_on_termination = true
    }
  }

  tag_specifications {
    resource_type = "instance"
    tags = merge(local.common_tags, {
      Name          = "${var.project_name}-gpu"
      AutoTerminate = "true"
    })
  }

  tags = local.common_tags
}

# =============================================================================
# Outputs
# =============================================================================

output "app_public_ip" {
  description = "Streamlit app public IP"
  value       = aws_eip.app.public_ip
}

output "streamlit_url" {
  description = "Streamlit URL (get the tunnel URL from instance logs)"
  value       = "Run: ssh ec2-user@${aws_eip.app.public_ip} 'journalctl -u cloudflared-tunnel -n 20' to find the https://xxxxx.trycloudflare.com URL"
}

output "database_endpoint" {
  description = "RDS endpoint"
  value       = aws_db_instance.main.endpoint
}

output "database_password_ssm" {
  description = "SSM parameter for DB password"
  value       = aws_ssm_parameter.db_password.name
}

output "s3_bucket" {
  description = "S3 bucket for all data"
  value       = aws_s3_bucket.data.id
}

output "gpu_launch_template" {
  description = "GPU launch template ID"
  value       = aws_launch_template.gpu.id
}

output "ssh_command" {
  description = "SSH command to connect to app server"
  value       = var.ssh_key_name != "" ? "ssh -i ~/.ssh/${var.ssh_key_name}.pem ec2-user@${aws_eip.app.public_ip}" : "Set ssh_key_name variable to enable SSH"
}

output "estimated_monthly_cost" {
  description = "Estimated monthly cost"
  value       = <<-EOT
    EC2 t3.small (app):     ~$15/month
    RDS db.t3.micro:        $0 (free tier) or ~$15/month
    S3 (50GB):              ~$2/month
    GPU (2 hours):          ~$1/month
    ----------------------------------------
    Total:                  ~$10-26/month

    Note: Free tier covers 750 hours/month of t3.micro for first year
  EOT
}

output "environment_variables" {
  description = "Environment variables for the app"
  sensitive   = true
  value       = <<-EOT
    PRISMATA_STORAGE_BACKEND=s3
    PRISMATA_S3_BUCKET=${aws_s3_bucket.data.id}
    PRISMATA_DB_HOST=${aws_db_instance.main.address}
    PRISMATA_DB_PORT=5432
    PRISMATA_DB_NAME=prismata
    PRISMATA_DB_USER=prismata
    PRISMATA_DB_PASSWORD=<from SSM: ${aws_ssm_parameter.db_password.name}>
    AWS_REGION=${var.aws_region}
  EOT
}
