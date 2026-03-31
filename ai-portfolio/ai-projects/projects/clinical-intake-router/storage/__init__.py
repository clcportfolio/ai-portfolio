"""
Storage package — Clinical Intake Router

Exposes the two storage utilities used by pipeline.py and app.py:
  - s3_client: file upload, download, listing, deletion, pre-signed URLs
  - db_client: Supabase/PostgreSQL insert, dedup check, deletion, NL2SQL query interface
"""

from storage.s3_client import upload_file, download_file, list_files, hash_file, get_presigned_url, delete_file
from storage.db_client import init_db, insert_submission, submission_exists, query_submissions, get_table_schema, get_recent_submissions, delete_submission

__all__ = [
    # S3
    "upload_file",
    "download_file",
    "list_files",
    "hash_file",
    "get_presigned_url",
    "delete_file",
    # DB
    "init_db",
    "insert_submission",
    "submission_exists",
    "query_submissions",
    "get_table_schema",
    "get_recent_submissions",
    "delete_submission",
]
