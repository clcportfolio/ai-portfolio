"""
provision_infra.py — Medical Triage Classifier
Sets up AWS S3 bucket structure for dataset storage, training checkpoints,
and model artifacts. Documents EC2 MLflow server manual setup steps.

Run:  python provision_infra.py
"""

import logging
import os
import sys

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(), override=True)

logger = logging.getLogger(__name__)

_S3_BUCKET = os.getenv("S3_BUCKET_NAME", "")
_AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# S3 prefix structure for this project
S3_PREFIXES = [
    "medical-triage-classifier/datasets/",
    "medical-triage-classifier/checkpoints/",
    "medical-triage-classifier/models/",
]


def _get_client():
    """Return a boto3 S3 client using env-var credentials."""
    return boto3.client(
        "s3",
        region_name=_AWS_REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )


def create_bucket_if_needed() -> str:
    """Create S3 bucket if it doesn't exist. Returns bucket name."""
    if not _S3_BUCKET:
        raise ValueError("S3_BUCKET_NAME environment variable is not set.")

    client = _get_client()
    try:
        client.head_bucket(Bucket=_S3_BUCKET)
        logger.info("Bucket %s already exists.", _S3_BUCKET)
    except ClientError as e:
        error_code = int(e.response["Error"]["Code"])
        if error_code == 404:
            logger.info("Creating bucket %s in %s...", _S3_BUCKET, _AWS_REGION)
            create_args = {"Bucket": _S3_BUCKET}
            if _AWS_REGION != "us-east-1":
                create_args["CreateBucketConfiguration"] = {
                    "LocationConstraint": _AWS_REGION
                }
            client.create_bucket(**create_args)
            # Enable versioning for checkpoint safety
            client.put_bucket_versioning(
                Bucket=_S3_BUCKET,
                VersioningConfiguration={"Status": "Enabled"},
            )
            logger.info("Bucket created with versioning enabled.")
        else:
            raise RuntimeError(f"S3 head_bucket failed: {e}") from e

    return _S3_BUCKET


def create_prefix_markers() -> list[str]:
    """Create zero-byte marker objects so prefixes are visible in the S3 console."""
    if not _S3_BUCKET:
        raise ValueError("S3_BUCKET_NAME environment variable is not set.")

    client = _get_client()
    created = []
    for prefix in S3_PREFIXES:
        try:
            client.put_object(Bucket=_S3_BUCKET, Key=prefix, Body=b"")
            created.append(prefix)
            logger.info("Created prefix marker: s3://%s/%s", _S3_BUCKET, prefix)
        except (BotoCoreError, ClientError) as e:
            logger.warning("Failed to create prefix %s: %s", prefix, e)

    return created


def print_ec2_mlflow_instructions():
    """Print manual setup steps for EC2 MLflow Tracking Server."""
    instructions = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                EC2 MLflow Tracking Server Setup (manual)                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  1. Launch EC2 instance (free tier: t2.micro, Amazon Linux 2023 AMI)       ║
║                                                                            ║
║  2. Security group: allow inbound TCP 5000 from your IP only               ║
║     (or use SSH tunnel for security)                                       ║
║                                                                            ║
║  3. SSH in and install:                                                    ║
║       sudo yum install python3-pip -y                                      ║
║       pip3 install mlflow boto3                                            ║
║                                                                            ║
║  4. Configure AWS credentials:                                             ║
║       aws configure  # enter IAM keys with S3 access                      ║
║                                                                            ║
║  5. Start MLflow server:                                                   ║
║       mlflow server \\                                                      ║
║         --host 0.0.0.0 \\                                                   ║
║         --port 5000 \\                                                      ║
║         --backend-store-uri sqlite:///mlflow.db \\                          ║
║         --default-artifact-root s3://{bucket}/medical-triage-classifier/   ║
║                                                                            ║
║  6. (Optional) Create systemd service for persistence:                     ║
║       sudo nano /etc/systemd/system/mlflow.service                         ║
║       # [Unit] Description=MLflow Tracking Server                          ║
║       # [Service] ExecStart=/usr/local/bin/mlflow server ...               ║
║       # [Install] WantedBy=multi-user.target                               ║
║       sudo systemctl enable --now mlflow                                   ║
║                                                                            ║
║  7. Update .env:                                                           ║
║       MLFLOW_TRACKING_URI=http://<ec2-public-ip>:5000                      ║
║                                                                            ║
║  Estimated cost: $0/mo on free tier (t2.micro, 750 hrs/mo for 12 months)   ║
╚══════════════════════════════════════════════════════════════════════════════╝
""".format(bucket=_S3_BUCKET or "<your-bucket-name>")
    print(instructions)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=== Medical Triage Classifier — Infrastructure Provisioning ===\n")

    if not _S3_BUCKET:
        print("ERROR: Set S3_BUCKET_NAME in .env before running.")
        sys.exit(1)

    # Step 1: S3 bucket
    print("1. Checking S3 bucket...")
    try:
        bucket = create_bucket_if_needed()
        print(f"   Bucket ready: {bucket}")
    except Exception as e:
        print(f"   FAILED: {e}")
        sys.exit(1)

    # Step 2: Prefix markers
    print("\n2. Creating S3 prefix structure...")
    prefixes = create_prefix_markers()
    for p in prefixes:
        print(f"   Created: s3://{_S3_BUCKET}/{p}")

    # Step 3: EC2 instructions
    print("\n3. EC2 MLflow Tracking Server setup:")
    print_ec2_mlflow_instructions()

    print("=== S3 provisioning complete. Follow EC2 steps above when ready. ===")
