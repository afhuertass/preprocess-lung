

""" firts shit on the cloud """

from google.cloud import storage

storage_client = storage.Client()

bucket_name = "ml-new-bucket"


bucket = storage_client.create_bucket(bucket_name)

print('Bucket {} created.'.format(bucket.name))
