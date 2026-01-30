"""GPU instance orchestration for on-demand training and inference.

Manages EC2 GPU instances that are spun up per job and terminated after completion.
This is optimized for light usage (~2 hours GPU/week) to minimize costs.
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from .config import AWSConfig


class GPUJobType(str, Enum):
    """Type of GPU job."""

    TRAINING = "training"
    INFERENCE = "inference"


class GPUJobStatus(str, Enum):
    """Status of GPU job."""

    PENDING = "pending"
    STARTING_INSTANCE = "starting_instance"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"


@dataclass
class GPUJobConfig:
    """Configuration for a GPU job."""

    job_type: GPUJobType
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Training-specific
    domain: str = ""
    labels_s3_prefix: str = ""
    videos_s3_prefix: str = ""
    output_model_s3_key: str = ""
    training_config: dict[str, Any] = field(default_factory=dict)

    # Inference-specific
    video_s3_key: str = ""
    model_s3_key: str = ""
    output_detection_s3_key: str = ""
    detection_threshold: float = 0.5

    # Instance configuration
    instance_type: str = "g4dn.xlarge"
    use_spot: bool = False
    max_runtime_minutes: int = 120  # Auto-terminate after this


@dataclass
class GPUJobResult:
    """Result of a GPU job."""

    job_id: str
    status: GPUJobStatus
    instance_id: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    runtime_seconds: int = 0
    output_s3_key: str | None = None
    error_message: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


class GPUOrchestrator:
    """Orchestrates on-demand GPU instances for training and inference.

    Usage:
        orchestrator = GPUOrchestrator(config)

        # Start a training job
        job_config = GPUJobConfig(
            job_type=GPUJobType.TRAINING,
            domain="cricket",
            labels_s3_prefix="labels/cricket/",
            output_model_s3_key="models/cricket/model_v1.pt",
        )
        result = orchestrator.run_job(job_config)

        # Or start inference
        job_config = GPUJobConfig(
            job_type=GPUJobType.INFERENCE,
            video_s3_key="videos/game1.mp4",
            model_s3_key="models/cricket/model_v1.pt",
        )
        result = orchestrator.run_job(job_config)
    """

    def __init__(self, config: AWSConfig):
        self.config = config
        self._ec2_client = None
        self._ssm_client = None
        self._cloudwatch_client = None

    @property
    def ec2_client(self):
        """Lazy initialization of EC2 client."""
        if self._ec2_client is None:
            import boto3

            self._ec2_client = boto3.client("ec2", region_name=self.config.region)
        return self._ec2_client

    @property
    def ssm_client(self):
        """Lazy initialization of SSM client."""
        if self._ssm_client is None:
            import boto3

            self._ssm_client = boto3.client("ssm", region_name=self.config.region)
        return self._ssm_client

    @property
    def cloudwatch_client(self):
        """Lazy initialization of CloudWatch client."""
        if self._cloudwatch_client is None:
            import boto3

            self._cloudwatch_client = boto3.client("logs", region_name=self.config.region)
        return self._cloudwatch_client

    def _get_user_data(self, job_config: GPUJobConfig) -> str:
        """Generate user data script for EC2 instance."""
        job_json = json.dumps({
            "job_id": job_config.job_id,
            "job_type": job_config.job_type.value,
            "domain": job_config.domain,
            "labels_s3_prefix": job_config.labels_s3_prefix,
            "videos_s3_prefix": job_config.videos_s3_prefix,
            "output_model_s3_key": job_config.output_model_s3_key,
            "video_s3_key": job_config.video_s3_key,
            "model_s3_key": job_config.model_s3_key,
            "output_detection_s3_key": job_config.output_detection_s3_key,
            "detection_threshold": job_config.detection_threshold,
            "training_config": job_config.training_config,
        })

        # Script to run on instance startup
        return f"""#!/bin/bash
set -e

# Log to CloudWatch
exec > >(tee /var/log/prismata-job.log | logger -t prismata-job -s 2>/dev/console) 2>&1

echo "Starting Prismata GPU job: {job_config.job_id}"

# Write job config
echo '{job_json}' > /tmp/job_config.json

# Activate conda environment (pre-installed in AMI)
source /opt/conda/etc/profile.d/conda.sh
conda activate prismata

# Pull latest code (optional, if not baked into AMI)
cd /opt/prismata
git pull origin main || true

# Run the job
python -m src.aws.job_runner /tmp/job_config.json

# Signal completion
echo "Job completed"

# Auto-terminate instance after job
if [ "$?" -eq 0 ]; then
    echo "Scheduling instance termination"
    sleep 30
    aws ec2 terminate-instances --instance-ids $(curl -s http://169.254.169.254/latest/meta-data/instance-id) --region {self.config.region}
fi
"""

    def start_instance(self, job_config: GPUJobConfig) -> str:
        """Start a GPU instance for the job.

        Returns:
            Instance ID of the started instance.
        """
        import base64

        user_data = self._get_user_data(job_config)
        user_data_b64 = base64.b64encode(user_data.encode()).decode()

        # Instance configuration
        instance_params = {
            "ImageId": self.config.gpu_ami_id,
            "InstanceType": job_config.instance_type or self.config.gpu_instance_type,
            "MinCount": 1,
            "MaxCount": 1,
            "KeyName": self.config.gpu_key_pair,
            "SecurityGroupIds": [self.config.gpu_security_group],
            "SubnetId": self.config.gpu_subnet_id,
            "IamInstanceProfile": {"Name": self.config.gpu_iam_instance_profile},
            "UserData": user_data_b64,
            "TagSpecifications": [
                {
                    "ResourceType": "instance",
                    "Tags": [
                        {"Key": "Name", "Value": f"prismata-gpu-{job_config.job_id[:8]}"},
                        {"Key": "Project", "Value": "Prismata"},
                        {"Key": "JobId", "Value": job_config.job_id},
                        {"Key": "JobType", "Value": job_config.job_type.value},
                        {"Key": "AutoTerminate", "Value": "true"},
                    ],
                }
            ],
            # Enable detailed monitoring for CloudWatch
            "Monitoring": {"Enabled": True},
            # Instance metadata options
            "MetadataOptions": {
                "HttpTokens": "required",  # IMDSv2
                "HttpEndpoint": "enabled",
            },
        }

        # Use spot instances if configured
        if job_config.use_spot or self.config.use_spot_instances:
            instance_params["InstanceMarketOptions"] = {
                "MarketType": "spot",
                "SpotOptions": {
                    "MaxPrice": self.config.spot_max_price,
                    "SpotInstanceType": "one-time",
                    "InstanceInterruptionBehavior": "terminate",
                },
            }

        response = self.ec2_client.run_instances(**instance_params)
        instance_id = response["Instances"][0]["InstanceId"]

        return instance_id

    def wait_for_instance_running(
        self, instance_id: str, timeout_seconds: int = 300
    ) -> bool:
        """Wait for instance to be in running state."""
        waiter = self.ec2_client.get_waiter("instance_running")
        try:
            waiter.wait(
                InstanceIds=[instance_id],
                WaiterConfig={"Delay": 10, "MaxAttempts": timeout_seconds // 10},
            )
            return True
        except Exception as e:
            print(f"Error waiting for instance: {e}")
            return False

    def get_instance_status(self, instance_id: str) -> dict[str, Any]:
        """Get current status of an instance."""
        response = self.ec2_client.describe_instances(InstanceIds=[instance_id])
        if response["Reservations"]:
            instance = response["Reservations"][0]["Instances"][0]
            return {
                "instance_id": instance_id,
                "state": instance["State"]["Name"],
                "launch_time": instance.get("LaunchTime"),
                "public_ip": instance.get("PublicIpAddress"),
                "private_ip": instance.get("PrivateIpAddress"),
            }
        return {"instance_id": instance_id, "state": "unknown"}

    def terminate_instance(self, instance_id: str) -> bool:
        """Terminate an instance."""
        try:
            self.ec2_client.terminate_instances(InstanceIds=[instance_id])
            return True
        except Exception as e:
            print(f"Error terminating instance: {e}")
            return False

    def run_command_on_instance(
        self, instance_id: str, command: str, timeout_seconds: int = 600
    ) -> dict[str, Any]:
        """Run a command on instance using SSM Run Command."""
        response = self.ssm_client.send_command(
            InstanceIds=[instance_id],
            DocumentName="AWS-RunShellScript",
            Parameters={"commands": [command]},
            TimeoutSeconds=timeout_seconds,
            CloudWatchOutputConfig={
                "CloudWatchLogGroupName": self.config.cloudwatch_log_group,
                "CloudWatchOutputEnabled": True,
            },
        )

        command_id = response["Command"]["CommandId"]

        # Wait for command to complete
        while True:
            result = self.ssm_client.get_command_invocation(
                CommandId=command_id,
                InstanceId=instance_id,
            )

            status = result["Status"]
            if status in ["Success", "Failed", "Cancelled", "TimedOut"]:
                return {
                    "status": status,
                    "stdout": result.get("StandardOutputContent", ""),
                    "stderr": result.get("StandardErrorContent", ""),
                    "exit_code": result.get("ResponseCode", -1),
                }

            time.sleep(5)

    def get_job_logs(self, job_id: str, limit: int = 100) -> list[str]:
        """Get CloudWatch logs for a job."""
        log_stream_name = f"job-{job_id}"
        try:
            response = self.cloudwatch_client.get_log_events(
                logGroupName=self.config.cloudwatch_log_group,
                logStreamName=log_stream_name,
                limit=limit,
                startFromHead=False,
            )
            return [event["message"] for event in response.get("events", [])]
        except Exception:
            return []

    def run_job(
        self,
        job_config: GPUJobConfig,
        progress_callback: Callable[[GPUJobStatus, str], None] | None = None,
        wait_for_completion: bool = True,
    ) -> GPUJobResult:
        """Run a GPU job.

        Args:
            job_config: Configuration for the job
            progress_callback: Optional callback for status updates
            wait_for_completion: If True, wait for job to complete

        Returns:
            GPUJobResult with job outcome
        """
        result = GPUJobResult(
            job_id=job_config.job_id,
            status=GPUJobStatus.PENDING,
            start_time=datetime.utcnow(),
        )

        def update_status(status: GPUJobStatus, message: str = ""):
            result.status = status
            if progress_callback:
                progress_callback(status, message)

        try:
            # Start instance
            update_status(GPUJobStatus.STARTING_INSTANCE, "Launching GPU instance")
            instance_id = self.start_instance(job_config)
            result.instance_id = instance_id

            # Wait for instance to be running
            if not self.wait_for_instance_running(instance_id):
                raise RuntimeError("Instance failed to start")

            update_status(GPUJobStatus.RUNNING, f"Instance {instance_id} is running")

            if not wait_for_completion:
                # Return early, caller can poll for status
                return result

            # Poll for job completion
            max_wait = job_config.max_runtime_minutes * 60
            start_wait = time.time()

            while time.time() - start_wait < max_wait:
                status = self.get_instance_status(instance_id)

                if status["state"] == "terminated":
                    # Instance terminated, check if job completed successfully
                    result.end_time = datetime.utcnow()
                    result.runtime_seconds = int(
                        (result.end_time - result.start_time).total_seconds()
                    )

                    # Try to get logs to determine success/failure
                    logs = self.get_job_logs(job_config.job_id)
                    if any("Job completed" in log for log in logs):
                        result.status = GPUJobStatus.COMPLETED
                        result.output_s3_key = (
                            job_config.output_model_s3_key
                            or job_config.output_detection_s3_key
                        )
                    else:
                        result.status = GPUJobStatus.FAILED
                        result.error_message = "Job terminated without completion"

                    return result

                if status["state"] == "stopped":
                    result.status = GPUJobStatus.FAILED
                    result.error_message = "Instance stopped unexpectedly"
                    self.terminate_instance(instance_id)
                    return result

                time.sleep(30)

            # Timeout reached
            result.status = GPUJobStatus.FAILED
            result.error_message = f"Job timed out after {job_config.max_runtime_minutes} minutes"
            self.terminate_instance(instance_id)

        except Exception as e:
            result.status = GPUJobStatus.FAILED
            result.error_message = str(e)
            if result.instance_id:
                self.terminate_instance(result.instance_id)

        result.end_time = datetime.utcnow()
        if result.start_time:
            result.runtime_seconds = int(
                (result.end_time - result.start_time).total_seconds()
            )

        return result

    def start_job_async(self, job_config: GPUJobConfig) -> GPUJobResult:
        """Start a job without waiting for completion.

        Returns immediately after instance is launched. Use get_job_status()
        to poll for completion.
        """
        return self.run_job(job_config, wait_for_completion=False)

    def list_running_jobs(self) -> list[dict[str, Any]]:
        """List all running GPU job instances."""
        response = self.ec2_client.describe_instances(
            Filters=[
                {"Name": "tag:Project", "Values": ["Prismata"]},
                {"Name": "tag:AutoTerminate", "Values": ["true"]},
                {"Name": "instance-state-name", "Values": ["pending", "running"]},
            ]
        )

        jobs = []
        for reservation in response.get("Reservations", []):
            for instance in reservation.get("Instances", []):
                tags = {t["Key"]: t["Value"] for t in instance.get("Tags", [])}
                jobs.append({
                    "instance_id": instance["InstanceId"],
                    "job_id": tags.get("JobId"),
                    "job_type": tags.get("JobType"),
                    "state": instance["State"]["Name"],
                    "launch_time": instance.get("LaunchTime"),
                    "instance_type": instance["InstanceType"],
                })

        return jobs

    def cleanup_stale_instances(self, max_age_hours: int = 4) -> int:
        """Terminate instances that have been running too long."""
        from datetime import timedelta

        running_jobs = self.list_running_jobs()
        terminated = 0
        now = datetime.utcnow()

        for job in running_jobs:
            launch_time = job.get("launch_time")
            if launch_time:
                # Make launch_time timezone-naive for comparison
                if launch_time.tzinfo:
                    launch_time = launch_time.replace(tzinfo=None)
                age = now - launch_time
                if age > timedelta(hours=max_age_hours):
                    self.terminate_instance(job["instance_id"])
                    terminated += 1

        return terminated
