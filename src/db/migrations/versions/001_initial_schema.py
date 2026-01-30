"""Initial schema for Prismata.

Revision ID: 001_initial
Revises:
Create Date: 2026-01-24

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create enum types
    video_status_enum = postgresql.ENUM(
        "uploaded", "processing", "ready", "error",
        name="videostatus",
        create_type=True,
    )
    job_status_enum = postgresql.ENUM(
        "pending", "running", "completed", "failed", "cancelled",
        name="jobstatus",
        create_type=True,
    )
    review_status_enum = postgresql.ENUM(
        "pending", "approved", "rejected",
        name="reviewstatus",
        create_type=True,
    )

    # Create jobs table first (referenced by others)
    op.create_table(
        "jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("job_type", sa.String(64), nullable=False),
        sa.Column("aws_job_id", sa.String(256)),
        sa.Column("aws_job_name", sa.String(256)),
        sa.Column("status", job_status_enum, server_default="pending"),
        sa.Column("progress", sa.Integer, server_default="0"),
        sa.Column("status_message", sa.Text),
        sa.Column("error_message", sa.Text),
        sa.Column("config", postgresql.JSONB),
        sa.Column("result", postgresql.JSONB),
        sa.Column("instance_type", sa.String(64)),
        sa.Column("runtime_seconds", sa.Integer),
        sa.Column("estimated_cost", sa.Numeric(10, 4)),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("started_at", sa.DateTime),
        sa.Column("completed_at", sa.DateTime),
    )
    op.create_index("idx_jobs_type_status", "jobs", ["job_type", "status"])
    op.create_index("idx_jobs_aws_job_id", "jobs", ["aws_job_id"])

    # Create videos table
    op.create_table(
        "videos",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("s3_key", sa.String(1024), nullable=False, unique=True),
        sa.Column("filename", sa.String(512), nullable=False),
        sa.Column("domain", sa.String(64)),
        sa.Column("width", sa.Integer),
        sa.Column("height", sa.Integer),
        sa.Column("fps", sa.Numeric(10, 4)),
        sa.Column("total_frames", sa.Integer),
        sa.Column("duration_seconds", sa.Numeric(12, 4)),
        sa.Column("file_size_bytes", sa.Integer),
        sa.Column("status", video_status_enum, server_default="uploaded"),
        sa.Column("error_message", sa.Text),
        sa.Column("uploaded_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("processed_at", sa.DateTime),
    )
    op.create_index("idx_videos_domain_status", "videos", ["domain", "status"])

    # Create labels table
    op.create_table(
        "labels",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "video_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("videos.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("event_id", sa.String(64), nullable=False),
        sa.Column("start_frame", sa.Integer, nullable=False),
        sa.Column("end_frame", sa.Integer, nullable=False),
        sa.Column("event_type", sa.String(64), server_default="default"),
        sa.Column("is_false_positive", sa.Boolean, server_default="false"),
        sa.Column("source", sa.String(64), server_default="manual"),
        sa.Column("confidence", sa.Numeric(5, 4)),
        sa.Column("notes", sa.Text),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index("idx_labels_video_id", "labels", ["video_id"])
    op.create_index("idx_labels_event_type", "labels", ["event_type"])

    # Create models table
    op.create_table(
        "models",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(256), nullable=False),
        sa.Column("domain", sa.String(64), nullable=False),
        sa.Column("version", sa.String(64)),
        sa.Column("s3_key", sa.String(1024), nullable=False, unique=True),
        sa.Column("config", postgresql.JSONB),
        sa.Column("metrics", postgresql.JSONB),
        sa.Column("val_f1", sa.Numeric(6, 4)),
        sa.Column("val_accuracy", sa.Numeric(6, 4)),
        sa.Column("val_precision", sa.Numeric(6, 4)),
        sa.Column("val_recall", sa.Numeric(6, 4)),
        sa.Column(
            "training_job_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("jobs.id"),
        ),
        sa.Column("num_epochs", sa.Integer),
        sa.Column("training_samples", sa.Integer),
        sa.Column("validation_samples", sa.Integer),
        sa.Column("is_production", sa.Boolean, server_default="false"),
        sa.Column("is_archived", sa.Boolean, server_default="false"),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("published_at", sa.DateTime),
    )
    op.create_index("idx_models_domain_production", "models", ["domain", "is_production"])

    # Create detections table
    op.create_table(
        "detections",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "video_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("videos.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "model_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("models.id"),
            nullable=False,
        ),
        sa.Column("threshold", sa.Numeric(5, 4), server_default="0.5"),
        sa.Column("merge_distance", sa.Numeric(10, 4)),
        sa.Column("num_events", sa.Integer, server_default="0"),
        sa.Column("s3_results_key", sa.String(1024)),
        sa.Column("status", job_status_enum, server_default="pending"),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("jobs.id")),
        sa.Column("error_message", sa.Text),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime),
    )
    op.create_index("idx_detections_video_model", "detections", ["video_id", "model_id"])
    op.create_index("idx_detections_status", "detections", ["status"])

    # Create detection_events table
    op.create_table(
        "detection_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "detection_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("detections.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("start_time", sa.Numeric(12, 4), nullable=False),
        sa.Column("end_time", sa.Numeric(12, 4), nullable=False),
        sa.Column("start_frame", sa.Integer),
        sa.Column("end_frame", sa.Integer),
        sa.Column("confidence", sa.Numeric(5, 4), nullable=False),
        sa.Column("review_status", review_status_enum, server_default="pending"),
        sa.Column("reviewed_at", sa.DateTime),
        sa.Column("reviewed_by", sa.String(256)),
        sa.Column("label_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("labels.id")),
        sa.Column("clip_s3_key", sa.String(1024)),
    )
    op.create_index("idx_detection_events_detection_id", "detection_events", ["detection_id"])
    op.create_index("idx_detection_events_review_status", "detection_events", ["review_status"])


def downgrade() -> None:
    op.drop_table("detection_events")
    op.drop_table("detections")
    op.drop_table("models")
    op.drop_table("labels")
    op.drop_table("videos")
    op.drop_table("jobs")

    # Drop enum types
    op.execute("DROP TYPE IF EXISTS reviewstatus")
    op.execute("DROP TYPE IF EXISTS jobstatus")
    op.execute("DROP TYPE IF EXISTS videostatus")
