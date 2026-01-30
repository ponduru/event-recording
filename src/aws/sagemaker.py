"""SageMaker integration for training and inference.

Provides an alternative to on-demand EC2 instances using SageMaker
managed infrastructure. Useful for larger training jobs or when
SageMaker's managed infrastructure is preferred.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from .config import AWSConfig


@dataclass
class TrainingJobConfig:
    """Configuration for SageMaker training job."""

    job_name: str = field(default_factory=lambda: f"prismata-train-{uuid.uuid4().hex[:8]}")
    domain: str = ""

    # Input data
    labels_s3_uri: str = ""
    videos_s3_uri: str = ""

    # Output
    output_s3_uri: str = ""

    # Hyperparameters
    hyperparameters: dict[str, str] = field(default_factory=dict)

    # Instance configuration
    instance_type: str = "ml.g4dn.xlarge"
    instance_count: int = 1
    volume_size_gb: int = 100

    # Training settings
    max_runtime_seconds: int = 3600 * 4  # 4 hours
    use_spot: bool = True
    max_wait_seconds: int = 3600 * 6  # 6 hours (if using spot)

    # Metrics to track
    metric_definitions: list[dict[str, str]] = field(default_factory=lambda: [
        {"Name": "train:loss", "Regex": "train_loss: ([0-9\\.]+)"},
        {"Name": "train:accuracy", "Regex": "train_accuracy: ([0-9\\.]+)"},
        {"Name": "val:loss", "Regex": "val_loss: ([0-9\\.]+)"},
        {"Name": "val:accuracy", "Regex": "val_accuracy: ([0-9\\.]+)"},
        {"Name": "val:f1", "Regex": "val_f1: ([0-9\\.]+)"},
    ])


@dataclass
class InferenceJobConfig:
    """Configuration for SageMaker processing/transform job."""

    job_name: str = field(default_factory=lambda: f"prismata-infer-{uuid.uuid4().hex[:8]}")

    # Model
    model_s3_uri: str = ""

    # Input video
    video_s3_uri: str = ""

    # Output
    output_s3_uri: str = ""

    # Detection settings
    threshold: float = 0.5

    # Instance configuration
    instance_type: str = "ml.g4dn.xlarge"
    instance_count: int = 1

    # Processing settings
    max_runtime_seconds: int = 3600  # 1 hour


@dataclass
class SageMakerJobResult:
    """Result of a SageMaker job."""

    job_name: str
    job_arn: str = ""
    status: str = "Unknown"
    start_time: datetime | None = None
    end_time: datetime | None = None
    model_artifact_s3_uri: str | None = None
    output_s3_uri: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    error_message: str | None = None


class SageMakerClient:
    """Client for SageMaker training and inference.

    Usage:
        client = SageMakerClient(config)

        # Start training
        job_config = TrainingJobConfig(
            domain="cricket",
            labels_s3_uri="s3://bucket/labels/",
            videos_s3_uri="s3://bucket/videos/",
            output_s3_uri="s3://bucket/models/",
        )
        job = client.start_training_job(job_config)

        # Check status
        result = client.get_training_job_status(job.job_name)

        # Run inference
        infer_config = InferenceJobConfig(
            model_s3_uri="s3://bucket/models/model.tar.gz",
            video_s3_uri="s3://bucket/videos/video.mp4",
            output_s3_uri="s3://bucket/detections/",
        )
        result = client.run_inference_job(infer_config)
    """

    def __init__(self, config: AWSConfig):
        self.config = config
        self._sagemaker_client = None
        self._s3_client = None

    @property
    def sagemaker_client(self):
        """Lazy initialization of SageMaker client."""
        if self._sagemaker_client is None:
            import boto3

            self._sagemaker_client = boto3.client(
                "sagemaker", region_name=self.config.region
            )
        return self._sagemaker_client

    @property
    def s3_client(self):
        """Lazy initialization of S3 client."""
        if self._s3_client is None:
            import boto3

            self._s3_client = boto3.client("s3", region_name=self.config.region)
        return self._s3_client

    def _get_default_hyperparameters(self, domain: str) -> dict[str, str]:
        """Get default hyperparameters for a domain."""
        return {
            "domain": domain,
            "epochs": "50",
            "batch_size": "8",
            "learning_rate": "0.0001",
            "window_size": "8",
            "target_fps": "10.0",
        }

    def start_training_job(self, job_config: TrainingJobConfig) -> SageMakerJobResult:
        """Start a SageMaker training job.

        Args:
            job_config: Training job configuration

        Returns:
            SageMakerJobResult with job details
        """
        # Merge default and custom hyperparameters
        hyperparameters = self._get_default_hyperparameters(job_config.domain)
        hyperparameters.update(job_config.hyperparameters)

        # Build training job request
        training_params = {
            "TrainingJobName": job_config.job_name,
            "AlgorithmSpecification": {
                "TrainingImage": self.config.sagemaker_training_image,
                "TrainingInputMode": "File",
                "MetricDefinitions": job_config.metric_definitions,
            },
            "RoleArn": self.config.sagemaker_role_arn,
            "InputDataConfig": [
                {
                    "ChannelName": "labels",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": job_config.labels_s3_uri,
                            "S3DataDistributionType": "FullyReplicated",
                        }
                    },
                },
                {
                    "ChannelName": "videos",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": job_config.videos_s3_uri,
                            "S3DataDistributionType": "FullyReplicated",
                        }
                    },
                },
            ],
            "OutputDataConfig": {
                "S3OutputPath": job_config.output_s3_uri,
            },
            "ResourceConfig": {
                "InstanceType": job_config.instance_type,
                "InstanceCount": job_config.instance_count,
                "VolumeSizeInGB": job_config.volume_size_gb,
            },
            "StoppingCondition": {
                "MaxRuntimeInSeconds": job_config.max_runtime_seconds,
            },
            "HyperParameters": hyperparameters,
            "Tags": [
                {"Key": "Project", "Value": "Prismata"},
                {"Key": "Domain", "Value": job_config.domain},
            ],
        }

        # Enable spot training if configured
        if job_config.use_spot:
            training_params["EnableManagedSpotTraining"] = True
            training_params["StoppingCondition"]["MaxWaitTimeInSeconds"] = (
                job_config.max_wait_seconds
            )

        # Start the job
        response = self.sagemaker_client.create_training_job(**training_params)

        return SageMakerJobResult(
            job_name=job_config.job_name,
            job_arn=response["TrainingJobArn"],
            status="InProgress",
            start_time=datetime.utcnow(),
        )

    def get_training_job_status(self, job_name: str) -> SageMakerJobResult:
        """Get status of a training job."""
        response = self.sagemaker_client.describe_training_job(
            TrainingJobName=job_name
        )

        result = SageMakerJobResult(
            job_name=job_name,
            job_arn=response.get("TrainingJobArn", ""),
            status=response.get("TrainingJobStatus", "Unknown"),
            start_time=response.get("TrainingStartTime"),
            end_time=response.get("TrainingEndTime"),
        )

        # Get model artifact location
        if response.get("ModelArtifacts"):
            result.model_artifact_s3_uri = response["ModelArtifacts"].get(
                "S3ModelArtifacts"
            )

        # Get metrics
        if response.get("FinalMetricDataList"):
            result.metrics = {
                m["MetricName"]: m["Value"]
                for m in response["FinalMetricDataList"]
            }

        # Get error message if failed
        if response.get("FailureReason"):
            result.error_message = response["FailureReason"]

        return result

    def wait_for_training_job(
        self,
        job_name: str,
        progress_callback: Callable[[str, dict], None] | None = None,
        poll_interval_seconds: int = 60,
    ) -> SageMakerJobResult:
        """Wait for training job to complete.

        Args:
            job_name: Name of the training job
            progress_callback: Optional callback for status updates
            poll_interval_seconds: How often to check status

        Returns:
            Final SageMakerJobResult
        """
        import time

        terminal_states = {"Completed", "Failed", "Stopped"}

        while True:
            result = self.get_training_job_status(job_name)

            if progress_callback:
                progress_callback(result.status, result.metrics)

            if result.status in terminal_states:
                return result

            time.sleep(poll_interval_seconds)

    def stop_training_job(self, job_name: str) -> bool:
        """Stop a training job."""
        try:
            self.sagemaker_client.stop_training_job(TrainingJobName=job_name)
            return True
        except Exception as e:
            print(f"Error stopping job: {e}")
            return False

    def run_inference_job(
        self, job_config: InferenceJobConfig
    ) -> SageMakerJobResult:
        """Run inference using SageMaker Processing job.

        For batch inference on videos, we use a Processing job rather than
        a real-time endpoint, as videos are processed asynchronously.
        """
        # Build processing job request
        processing_params = {
            "ProcessingJobName": job_config.job_name,
            "ProcessingResources": {
                "ClusterConfig": {
                    "InstanceCount": job_config.instance_count,
                    "InstanceType": job_config.instance_type,
                    "VolumeSizeInGB": 50,
                }
            },
            "StoppingCondition": {
                "MaxRuntimeInSeconds": job_config.max_runtime_seconds
            },
            "AppSpecification": {
                "ImageUri": self.config.sagemaker_inference_image,
                "ContainerEntrypoint": ["python", "-m", "src.aws.inference_runner"],
                "ContainerArguments": [
                    "--model-uri",
                    job_config.model_s3_uri,
                    "--threshold",
                    str(job_config.threshold),
                ],
            },
            "ProcessingInputs": [
                {
                    "InputName": "video",
                    "S3Input": {
                        "S3Uri": job_config.video_s3_uri,
                        "LocalPath": "/opt/ml/processing/input/video",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File",
                    },
                },
                {
                    "InputName": "model",
                    "S3Input": {
                        "S3Uri": job_config.model_s3_uri,
                        "LocalPath": "/opt/ml/processing/input/model",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File",
                    },
                },
            ],
            "ProcessingOutputConfig": {
                "Outputs": [
                    {
                        "OutputName": "detections",
                        "S3Output": {
                            "S3Uri": job_config.output_s3_uri,
                            "LocalPath": "/opt/ml/processing/output",
                            "S3UploadMode": "EndOfJob",
                        },
                    }
                ]
            },
            "RoleArn": self.config.sagemaker_role_arn,
            "Tags": [
                {"Key": "Project", "Value": "Prismata"},
            ],
        }

        response = self.sagemaker_client.create_processing_job(**processing_params)

        return SageMakerJobResult(
            job_name=job_config.job_name,
            job_arn=response.get("ProcessingJobArn", ""),
            status="InProgress",
            start_time=datetime.utcnow(),
            output_s3_uri=job_config.output_s3_uri,
        )

    def get_processing_job_status(self, job_name: str) -> SageMakerJobResult:
        """Get status of a processing job."""
        response = self.sagemaker_client.describe_processing_job(
            ProcessingJobName=job_name
        )

        result = SageMakerJobResult(
            job_name=job_name,
            job_arn=response.get("ProcessingJobArn", ""),
            status=response.get("ProcessingJobStatus", "Unknown"),
            start_time=response.get("ProcessingStartTime"),
            end_time=response.get("ProcessingEndTime"),
        )

        # Get output location
        outputs = response.get("ProcessingOutputConfig", {}).get("Outputs", [])
        if outputs:
            result.output_s3_uri = outputs[0].get("S3Output", {}).get("S3Uri")

        # Get error message if failed
        if response.get("FailureReason"):
            result.error_message = response["FailureReason"]

        return result

    def list_training_jobs(
        self,
        status_filter: str | None = None,
        max_results: int = 20,
    ) -> list[SageMakerJobResult]:
        """List recent training jobs."""
        params = {
            "SortBy": "CreationTime",
            "SortOrder": "Descending",
            "MaxResults": max_results,
            "NameContains": "prismata",
        }

        if status_filter:
            params["StatusEquals"] = status_filter

        response = self.sagemaker_client.list_training_jobs(**params)

        results = []
        for job in response.get("TrainingJobSummaries", []):
            results.append(
                SageMakerJobResult(
                    job_name=job["TrainingJobName"],
                    job_arn=job["TrainingJobArn"],
                    status=job["TrainingJobStatus"],
                    start_time=job.get("TrainingStartTime"),
                    end_time=job.get("TrainingEndTime"),
                )
            )

        return results
