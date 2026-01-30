#!/bin/bash
# Initialize S3 bucket in LocalStack (single-bucket mode)

echo "Creating S3 bucket..."
awslocal s3 mb s3://prismata-data-local

echo "Configuring CORS for video streaming..."
awslocal s3api put-bucket-cors --bucket prismata-data-local --cors-configuration '{
  "CORSRules": [
    {
      "AllowedHeaders": ["*"],
      "AllowedMethods": ["GET", "HEAD"],
      "AllowedOrigins": ["*"],
      "ExposeHeaders": ["ETag", "Content-Length"],
      "MaxAgeSeconds": 3600
    }
  ]
}'

echo "Done. Bucket: prismata-data-local"
echo "  videos/    -> video files"
echo "  labels/    -> label JSON files"
echo "  models/    -> model checkpoints"
echo "  detections/ -> detection results"
