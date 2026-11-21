"""Is horse_racing.db versioned in S3 (for the shadow price on the live run's database; shadow-m1-1121 diagnosis)?"""
import boto3

s3 = boto3.client("s3")
for b in ("horseracingresults",):
    try:
        print(b, "versioning:", s3.get_bucket_versioning(Bucket=b).get("Status", "never enabled"))
    except Exception as e:  # noqa: BLE001
        print(b, "versioning: not readable", type(e).__name__)
    try:
        vs = s3.list_object_versions(Bucket=b, Prefix="horse_racing.db", MaxKeys=20).get("Versions", [])
        for v in vs[:12]:
            print(f"  {v['Key']} {v['LastModified']} {v['Size']:,} latest={v['IsLatest']} id={v['VersionId'][:12]}")
    except Exception as e:  # noqa: BLE001
        print("  versions: not readable", type(e).__name__, e)
