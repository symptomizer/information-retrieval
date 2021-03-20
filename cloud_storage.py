from google.cloud import storage
import os.path
import os

def test_file_exists():
    print("Key File Exists")
    print(os.path.isfile("keyfile.json"))

def check_if_exists(bucket_name, file_name):
    storage_client = storage.Client.from_service_account_json('keyfile.json')
    bucket = storage_client.bucket(bucket_name)
    return file_name in [blb.name for blb in list(storage_client.list_blobs(bucket))]


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client.from_service_account_json('keyfile.json')
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )

# upload_blob("symptomizer_indices_bucket-1", "hello.txt", "hello.txt")
# print(check_if_exists("symptomizer_indices_bucket-1", "hello.txt"))
# download_blob("symptomizer_indices_bucket-1", "hello.txt", "downloaded.txt")


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client.from_service_account_json('keyfile.json')

    bucket = storage_client.bucket(bucket_name)

    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Blob {} downloaded to {}.".format(
            source_blob_name, destination_file_name
        )
    )

def pull_indices():
    # Checks if PULL_INDS environment variable is present, and calls pull function
    if os.environ.get('PULL_INDS') != None:
        download_blob("symptomizer_indices_bucket-1", "tfidf.index", "models/tfidf.index")
        download_blob("symptomizer_indices_bucket-1", "bert.index", "models/bert.index")
        download_blob("symptomizer_indices_bucket-1", "ids.joblib", "models/ids.joblib")
