"""AWS configuration for Prismata."""

import os
from dataclasses import dataclass, field


@dataclass
class AWSConfig:
    """Configuration for AWS services."""

    # Region
    region: str = "us-east-1"

    # S3 buckets
    s3_bucket_videos: str = ""
    s3_bucket_labels: str = ""
    s3_bucket_models: str = ""
    s3_bucket_detections: str = ""

    # EC2 GPU configuration
    gpu_instance_type: str = "g4dn.xlarge"
    gpu_ami_id: str = ""  # Custom AMI with PyTorch, CUDA, app code
    gpu_key_pair: str = ""
    gpu_security_group: str = ""
    gpu_subnet_id: str = ""
    gpu_iam_instance_profile: str = ""

    # Spot instance configuration
    use_spot_instances: bool = False  # Use spot for training
    spot_max_price: str = "0.50"  # Max price per hour

    # Auto-termination
    auto_terminate_after_minutes: int = 30  # Terminate idle instances

    # SageMaker configuration (optional)
    sagemaker_role_arn: str = ""
    sagemaker_training_image: str = ""
    sagemaker_inference_image: str = ""

    # Lambda configuration
    lambda_orchestrator_arn: str = ""

    # Redis/ElastiCache
    redis_host: str = ""
    redis_port: int = 6379

    # CloudWatch
    cloudwatch_log_group: str = "/prismata/jobs"

    # VPC configuration
    vpc_id: str = ""

    # Tags
    tags: dict[str, str] = field(default_factory=lambda: {"Project": "Prismata"})

    @classmethod
    def from_env(cls) -> "AWSConfig":
        """Create configuration from environment variables."""
        return cls(
            region=os.getenv("AWS_REGION", "us-east-1"),
            s3_bucket_videos=os.getenv("PRISMATA_S3_BUCKET_VIDEOS", ""),
            s3_bucket_labels=os.getenv("PRISMATA_S3_BUCKET_LABELS", ""),
            s3_bucket_models=os.getenv("PRISMATA_S3_BUCKET_MODELS", ""),
            s3_bucket_detections=os.getenv("PRISMATA_S3_BUCKET_DETECTIONS", ""),
            gpu_instance_type=os.getenv("PRISMATA_GPU_INSTANCE_TYPE", "g4dn.xlarge"),
            gpu_ami_id=os.getenv("PRISMATA_GPU_AMI_ID", ""),
            gpu_key_pair=os.getenv("PRISMATA_GPU_KEY_PAIR", ""),
            gpu_security_group=os.getenv("PRISMATA_GPU_SECURITY_GROUP", ""),
            gpu_subnet_id=os.getenv("PRISMATA_GPU_SUBNET_ID", ""),
            gpu_iam_instance_profile=os.getenv("PRISMATA_GPU_IAM_PROFILE", ""),
            use_spot_instances=os.getenv("PRISMATA_USE_SPOT", "false").lower() == "true",
            spot_max_price=os.getenv("PRISMATA_SPOT_MAX_PRICE", "0.50"),
            auto_terminate_after_minutes=int(os.getenv("PRISMATA_AUTO_TERMINATE_MINUTES", "30")),
            sagemaker_role_arn=os.getenv("PRISMATA_SAGEMAKER_ROLE_ARN", ""),
            sagemaker_training_image=os.getenv("PRISMATA_SAGEMAKER_TRAINING_IMAGE", ""),
            sagemaker_inference_image=os.getenv("PRISMATA_SAGEMAKER_INFERENCE_IMAGE", ""),
            lambda_orchestrator_arn=os.getenv("PRISMATA_LAMBDA_ORCHESTRATOR_ARN", ""),
            redis_host=os.getenv("PRISMATA_REDIS_HOST", ""),
            redis_port=int(os.getenv("PRISMATA_REDIS_PORT", "6379")),
            cloudwatch_log_group=os.getenv("PRISMATA_CLOUDWATCH_LOG_GROUP", "/prismata/jobs"),
            vpc_id=os.getenv("PRISMATA_VPC_ID", ""),
        )

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []

        if not self.region:
            errors.append("AWS region is required")

        # Check S3 buckets
        if not self.s3_bucket_videos:
            errors.append("S3 bucket for videos is required")
        if not self.s3_bucket_models:
            errors.append("S3 bucket for models is required")

        # Check GPU configuration
        if not self.gpu_ami_id:
            errors.append("GPU AMI ID is required for EC2 training")
        if not self.gpu_subnet_id:
            errors.append("GPU subnet ID is required")
        if not self.gpu_security_group:
            errors.append("GPU security group is required")
        if not self.gpu_iam_instance_profile:
            errors.append("GPU IAM instance profile is required")

        return errors

    def is_configured(self) -> bool:
        """Check if AWS is properly configured."""
        return len(self.validate()) == 0
