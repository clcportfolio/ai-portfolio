"""
S3 Client — Clinical Intake Router
Handles file upload, listing, retrieval, and existence checks against AWS S3.

All functions are stateless — they read credentials from environment variables
on each call via boto3's default credential chain (env vars → ~/.aws → IAM role).

SHA-256 hashing is done here to keep the storage layer self-contained. The hash
is returned from upload_file() and stored in Postgres by db_client.py so that
duplicate detection can be done before making any S3 API call.
"""

import hashlib
import io
import logging
import os
from datetime import datetime
from typing import Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(), override=True)

logger = logging.getLogger(__name__)

_S3_BUCKET = os.getenv("S3_BUCKET_NAME", "")
_AWS_REGION = os.getenv("AWS_REGION", "us-east-1")


def _get_client():
    """Return a boto3 S3 client using env-var credentials."""
    return boto3.client(
        "s3",
        region_name=_AWS_REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )


def hash_file(file_bytes: bytes) -> str:
    """Return the SHA-256 hex digest of file_bytes."""
    return hashlib.sha256(file_bytes).hexdigest()


def upload_file(
    file_bytes: bytes,
    original_filename: str,
    content_type: str = "application/octet-stream",
) -> dict:
    """
    Upload a file to S3 under a timestamped key.

    Key format: intakes/<YYYY-MM-DD>/<sha256>_<original_filename>
    This makes files browsable by date in the S3 console and guarantees
    the key is unique even if the same filename is uploaded twice.

    Args:
        file_bytes: Raw file content.
        original_filename: Original name of the file (used in the S3 key).
        content_type: MIME type, e.g. "application/pdf" or "text/plain".

    Returns:
        dict with keys: s3_key, s3_bucket, file_hash, file_size_bytes, uploaded_at

    Raises:
        ValueError: If S3_BUCKET_NAME env var is not set.
        RuntimeError: On any S3 upload failure.
    """
    if not _S3_BUCKET:
        raise ValueError("S3_BUCKET_NAME environment variable is not set.")

    file_hash = hash_file(file_bytes)
    date_prefix = datetime.utcnow().strftime("%Y-%m-%d")
    s3_key = f"intakes/{date_prefix}/{file_hash[:12]}_{original_filename}"

    try:
        client = _get_client()
        client.upload_fileobj(
            io.BytesIO(file_bytes),
            _S3_BUCKET,
            s3_key,
            ExtraArgs={"ContentType": content_type},
        )
        logger.info("Uploaded %s to s3://%s/%s", original_filename, _S3_BUCKET, s3_key)
    except (BotoCoreError, ClientError) as e:
        logger.error("S3 upload failed: %s", e)
        raise RuntimeError(f"S3 upload failed: {e}") from e

    return {
        "s3_key": s3_key,
        "s3_bucket": _S3_BUCKET,
        "file_hash": file_hash,
        "file_size_bytes": len(file_bytes),
        "uploaded_at": datetime.utcnow().isoformat(),
    }


def download_file(s3_key: str) -> bytes:
    """
    Download a file from S3 and return its raw bytes.

    Args:
        s3_key: The S3 object key (as stored in the database).

    Returns:
        Raw file bytes.

    Raises:
        ValueError: If S3_BUCKET_NAME env var is not set.
        RuntimeError: If the file cannot be retrieved.
    """
    if not _S3_BUCKET:
        raise ValueError("S3_BUCKET_NAME environment variable is not set.")

    try:
        client = _get_client()
        response = client.get_object(Bucket=_S3_BUCKET, Key=s3_key)
        return response["Body"].read()
    except (BotoCoreError, ClientError) as e:
        logger.error("S3 download failed for key %s: %s", s3_key, e)
        raise RuntimeError(f"S3 download failed: {e}") from e


def list_files(prefix: str = "intakes/", max_keys: int = 200) -> list[dict]:
    """
    List files in the S3 bucket under the given prefix.

    Args:
        prefix: S3 key prefix to filter by. Default lists all intake files.
        max_keys: Maximum number of results to return.

    Returns:
        List of dicts, each with: s3_key, file_size_bytes, last_modified, filename.
        Sorted by last_modified descending (newest first).

    Raises:
        ValueError: If S3_BUCKET_NAME env var is not set.
        RuntimeError: On S3 list failure.
    """
    if not _S3_BUCKET:
        raise ValueError("S3_BUCKET_NAME environment variable is not set.")

    try:
        client = _get_client()
        response = client.list_objects_v2(
            Bucket=_S3_BUCKET,
            Prefix=prefix,
            MaxKeys=max_keys,
        )
    except (BotoCoreError, ClientError) as e:
        logger.error("S3 list failed: %s", e)
        raise RuntimeError(f"S3 list failed: {e}") from e

    contents = response.get("Contents", [])
    files = [
        {
            "s3_key": obj["Key"],
            "file_size_bytes": obj["Size"],
            "last_modified": obj["LastModified"].isoformat(),
            # Extract just the filename from the key for display
            "filename": obj["Key"].split("/")[-1],
        }
        for obj in contents
        if obj["Size"] > 0  # skip any zero-byte placeholder objects
    ]
    return sorted(files, key=lambda x: x["last_modified"], reverse=True)


def delete_file(s3_key: str) -> None:
    """
    Permanently delete a file from S3 by its object key.

    Args:
        s3_key: The S3 object key to delete (as stored in the database).

    Raises:
        ValueError: If S3_BUCKET_NAME env var is not set.
        RuntimeError: If the deletion fails.
    """
    if not _S3_BUCKET:
        raise ValueError("S3_BUCKET_NAME environment variable is not set.")

    try:
        client = _get_client()
        client.delete_object(Bucket=_S3_BUCKET, Key=s3_key)
        logger.info("Deleted s3://%s/%s", _S3_BUCKET, s3_key)
    except (BotoCoreError, ClientError) as e:
        logger.error("S3 delete failed for key %s: %s", s3_key, e)
        raise RuntimeError(f"S3 delete failed: {e}") from e


def get_presigned_url(s3_key: str, expiry_seconds: int = 3600) -> str:
    """
    Generate a pre-signed URL for temporary direct access to a file.

    Useful for allowing the Streamlit app to offer a download link without
    proxying the file bytes through the app server.

    Args:
        s3_key: The S3 object key.
        expiry_seconds: How long the URL is valid. Default 1 hour.

    Returns:
        Pre-signed HTTPS URL string.
    """
    if not _S3_BUCKET:
        raise ValueError("S3_BUCKET_NAME environment variable is not set.")

    client = _get_client()
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": _S3_BUCKET, "Key": s3_key},
        ExpiresIn=expiry_seconds,
    )


if __name__ == "__main__":
    """
    Smoke test — validates S3 connectivity and round-trip upload/download/list.
    Requires AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and S3_BUCKET_NAME in .env.
    """
    import json

    print("=== S3 Client Smoke Test ===\n")

    # Upload a small test file
    test_content = b"Clinical intake router S3 smoke test file."
    test_filename = "smoke_test.txt"

    print("1. Uploading test file...")
    upload_result = upload_file(test_content, test_filename, content_type="text/plain")
    print(json.dumps(upload_result, indent=2))

    # List files to confirm it appears
    print("\n2. Listing files under 'intakes/'...")
    files = list_files()
    print(f"   Found {len(files)} file(s). Most recent:")
    for f in files[:3]:
        print(f"   - {f['filename']} ({f['file_size_bytes']} bytes) @ {f['last_modified']}")

    # Download and verify round-trip
    print(f"\n3. Downloading {upload_result['s3_key']}...")
    downloaded = download_file(upload_result["s3_key"])
    assert downloaded == test_content, "Round-trip content mismatch!"
    print("   Content matches original. Round-trip OK.")

    # Pre-signed URL
    print("\n4. Generating pre-signed URL...")
    url = get_presigned_url(upload_result["s3_key"], expiry_seconds=300)
    print(f"   URL (5-min expiry): {url[:80]}...")

    print("\n=== All checks passed ===")
