from datetime import datetime, timedelta

import requests
import time

from requests import HTTPError
from datetime import datetime, timedelta

class WatsonStudioJobsManager:
    """
    A class to manage IBM Watson Studio jobs using the IBM Cloud API.

    Attributes:
        api_key (str): API key for IBM Cloud IAM authentication.
        project_id (str): ID of the IBM Cloud project.
        token (str): IAM access token.
        headers (dict): HTTP headers for API requests.
        params (dict): Default parameters for API requests.
        base_url (str): Base URL for IBM Cloud job management.
    """

    def __init__(self, api_key, project_id):
        """
        Initialize WatsonStudioJobsManager with API key and project ID.

        Args:
            api_key (str): IBM Cloud API key.
            project_id (str): Project ID.
        """
        self.api_key = api_key
        self.token_updated_at = None
        self.token = None
        self.headers = None
        self.update_token_if_needed()
        self.params = {"project_id": project_id}
        self.base_url = "https://api.dataplatform.cloud.ibm.com/v2/jobs"

    def update_token_if_needed(self):
        if self.token is None or datetime.now() - self.token_updated_at > timedelta(minutes=10):
            print(f'updating token')
            self.token = self._get_iam_token()
            self.token_updated_at = datetime.now()
            self.headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json"
            }

    def _get_iam_token(self):
        """
        Retrieve IAM token from IBM Cloud.

        Returns:
            str: IAM access token.
        """
        auth_url = "https://iam.cloud.ibm.com/identity/token"
        response = requests.post(auth_url, data={
            "apikey": self.api_key,
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey"
        })
        self.raise_for_status(response)
        return response.json()["access_token"]

    def raise_for_status(self, response):
        """Raises :class:`HTTPError`, if one occurred."""

        http_error_msg = ""
        if isinstance(response.reason, bytes):
            # Attempt to decode utf-8 first for localized strings; fall back to iso-8859-1
            try:
                reason = response.reason.decode("utf-8")
            except UnicodeDecodeError:
                reason = response.reason.decode("iso-8859-1")
        else:
            reason = response.reason

        if 400 <= response.status_code < 500:
            http_error_msg = (
                f"{response.status_code} Client Error: {reason} for url: {response.url}"
            )

        elif 500 <= response.status_code < 600:
            http_error_msg = (
                f"{response.status_code} Server Error: {reason} for url: {response.url}"
            )

        # Append response text if it exists
        if http_error_msg and hasattr(response, 'text'):
            http_error_msg += f"\nResponse text: {response.text}"

        if http_error_msg:
            raise HTTPError(http_error_msg, response=response)

    def create_job(self, job_name, notebook_asset_ref, env_variables:list[str]=None):
        """
        Create a new Watson Studio job.

        Args:
            job_name (str): Name of the job to create.
            notebook_asset_ref (str): Reference to the notebook asset.

        Returns:
            str: Asset ID of the created job.
        """
        self.update_token_if_needed()
        job_data = {
            "job": {
                "name": job_name,
                "asset_ref": notebook_asset_ref,
            }
        }

        if env_variables:
            job_data['job']['configuration'] = {'env_variables': env_variables}

        response = requests.post(self.base_url, headers=self.headers, json=job_data, params=self.params)
        self.raise_for_status(response)
        return response.json()['asset_id']

    def start_run(self, job_id):
        """
        Start a run of an existing job.

        Args:
            job_id (str): ID of the job to run.

        Returns:
            str: Asset ID of the job run.
        """
        self.update_token_if_needed()
        url = f"{self.base_url}/{job_id}/runs"
        response = requests.post(url, headers=self.headers, params=self.params)
        self.raise_for_status(response)
        return response.json()['metadata']['asset_id']

    def get_all_jobs(self):
        """
        Retrieve all jobs in the project.

        Returns:
            list: List of jobs with their details.
        """
        self.update_token_if_needed()
        response = requests.get(self.base_url, headers=self.headers, params=self.params)
        self.raise_for_status(response)
        return response.json()['results']

    def get_last_job_id(self):
        """
        Retrieve the last job ID created in the project.

        Returns:
            str: Last job ID.
        """
        self.update_token_if_needed()
        all_jobs = self.get_all_jobs()
        last_job_id = all_jobs[0]['metadata']['asset_id']
        return last_job_id

    def get_job_id_by_name(self, job_name):
        """
        Retrieve a job ID by its name.

        Args:
            job_name (str): Name of the job to search for.

        Returns:
            str: Job ID.

        Raises:
            Exception: If no job with the specified name is found.
        """
        self.update_token_if_needed()
        all_jobs = self.get_all_jobs()
        jobs_with_name = [j for j in all_jobs if j['metadata']['name'] == job_name]
        if len(jobs_with_name) >= 1:
            job = jobs_with_name[-1]
            job_id = job['metadata']['asset_id']
            return job_id
        raise Exception(f'Job: {job_name} was not found')

    def get_job_status(self, job_id):
        """
        Retrieve the status of a job.

        Args:
            job_id (str): ID of the job.

        Returns:
            dict: Job status details.
        """
        self.update_token_if_needed()
        url = f"{self.base_url}/{job_id}"
        response = requests.get(url, headers=self.headers, params=self.params)
        self.raise_for_status(response)
        return response.json()

    def get_job_runs(self, job_id, limit=None):
        """
        Retrieve job runs, with an option to limit the number of runs.

        Args:
            job_id (str): ID of the job.
            limit (int, optional): Limit the number of job runs to retrieve.

        Returns:
            list: List of job runs.
        """
        self.update_token_if_needed()
        url = f"{self.base_url}/{job_id}/runs"
        params = self.params
        if limit:
            params = self.params.copy()
            params['limit'] = limit
        response = requests.get(url, headers=self.headers, params=params)
        self.raise_for_status(response)
        return response.json()['results']

    def get_last_run(self, job_id):
        """
        Retrieve the last run of a job.

        Args:
            job_id (str): ID of the job.

        Returns:
            dict: Details of the last job run.
        """
        self.update_token_if_needed()
        results = self.get_job_runs(job_id=job_id, limit=1)
        return results[0]

    def get_last_run_id(self, job_id):
        """
        Retrieve the ID of the last job run.

        Args:
            job_id (str): ID of the job.

        Returns:
            str: ID of the last job run.
        """
        self.update_token_if_needed()
        return self.get_last_run(job_id=job_id)['metadata']['asset_id']

    def get_run(self, job_id, run_id):
        """
        Retrieve details of a specific job run.

        Args:
            job_id (str): ID of the job.
            run_id (str): ID of the run.

        Returns:
            dict: Details of the job run.
        """
        self.update_token_if_needed()
        url = f"{self.base_url}/{job_id}/runs/{run_id}"
        response = requests.get(url, headers=self.headers, params=self.params)
        self.raise_for_status(response)
        return response.json()

    def get_run_state(self, job_id, run_id):
        """
        Retrieve the state of a specific job run.

        Args:
            job_id (str): ID of the job.
            run_id (str): ID of the run.

        Returns:
            str: State of the job run.
        """
        self.update_token_if_needed()
        return self.get_run(job_id, run_id)['entity']['job_run']['state'].lower()

    def cancel_run(self, job_id, run_id):
        """
        Cancel a specific job run.

        Args:
            job_id (str): ID of the job.
            run_id (str): ID of the run.

        Returns:
            bool: True if cancellation was successful, False otherwise.
        """
        self.update_token_if_needed()
        url = f"{self.base_url}/{job_id}/runs/{run_id}/cancel"
        response = requests.post(url, headers=self.headers, params=self.params)
        self.raise_for_status(response)
        return response.status_code == 200

    def delete_run(self, job_id, run_id):
        """
        Delete a specific job run.

        Args:
            job_id (str): ID of the job.
            run_id (str): ID of the run.

        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        self.update_token_if_needed()
        url = f"{self.base_url}/{job_id}/runs/{run_id}"
        response = requests.delete(url, headers=self.headers, params=self.params)
        self.raise_for_status(response)
        return response.status_code == 204

    def delete_job(self, job_id):
        """
        Delete a job.

        Args:
            job_id (str): ID of the job.

        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        self.update_token_if_needed()
        url = f"{self.base_url}/{job_id}"
        response = requests.delete(url, headers=self.headers, params=self.params)
        self.raise_for_status(response)
        return response.status_code == 204

    def wait_for_run(self, job_id, run_id, interval=30):
        """
        Wait for a job run to complete, polling at regular intervals.

        Args:
            job_id (str): ID of the job.
            run_id (str): ID of the run.
            interval (int, optional): Time in seconds between status checks.

        Returns:
            str: Final state of the job run.
        """
        self.update_token_if_needed()
        while True:
            run_state = self.get_run_state(job_id=job_id, run_id=run_id)
            print(f"Job {job_id}, run: {run_id}, state: {run_state}")
            if run_state in ["completed", "failed", "canceled"]:
                return run_state
            time.sleep(interval)

    def get_run_logs(self, job_id, run_id):
        """
        Retrieve logs of a specific job run.

        Args:
            job_id (str): ID of the job.
            run_id (str): ID of the run.

        Returns:
            list: List of log entries.
        """
        self.update_token_if_needed()
        url = f"{self.base_url}/{job_id}/runs/{run_id}/logs"
        response = requests.get(url, headers=self.headers, params=self.params)
        self.raise_for_status(response)
        return response.json()['results']

    def get_all_envs(self):
        """
        Retrieves all available runtime environments in the Watson Studio project.

        Returns:
            list: List of environment details.
        """
        self.update_token_if_needed()
        url = "https://api.dataplatform.cloud.ibm.com/v2/environments"
        response = requests.get(url, headers=self.headers, params=self.params)
        self.raise_for_status(response)
        return response.json().get('resources', [])


# pip install ibm-cos-sdk
import ibm_boto3
from ibm_botocore.client import Config
import os


class CosFilesManager:
    def __init__(self, api_key, service_instance_id, endpoint_url, bucket_name):
        self.api_key = api_key
        self.service_instance_id = service_instance_id
        self.endpoint_url = endpoint_url
        self.bucket_name = bucket_name

        # Initialize the COS client
        self.cos_client = ibm_boto3.client(
            's3',
            ibm_api_key_id=self.api_key,
            ibm_service_instance_id=self.service_instance_id,
            config=Config(signature_version='oauth'),
            endpoint_url=self.endpoint_url
        )

    def upload_file(self, file_path, object_name=None):
        """Uploads a single file to the specified bucket in COS."""
        if object_name is None:
            object_name = os.path.basename(file_path)

        try:
            with open(file_path, "rb") as f:
                self.cos_client.upload_fileobj(f, self.bucket_name, object_name)
            print(f"File '{file_path}' uploaded as '{object_name}'.")
        except Exception as e:
            print(f"Failed to upload file: {e}")

    def upload_folder(self, folder_path, prefix=''):
        """Uploads all files in a folder to COS under a specified prefix."""
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                object_name = f"{prefix}/{os.path.relpath(file_path, folder_path)}"
                self.upload_file(file_path, object_name)

    def download_file(self, object_name, download_path):
        """Downloads a single file from COS."""
        try:
            os.makedirs(os.path.dirname(download_path), exist_ok=True)
            with open(download_path, 'wb') as f:
                self.cos_client.download_fileobj(self.bucket_name, object_name, f)
            print(f"File '{object_name}' downloaded to '{download_path}'.")
        except Exception as e:
            print(f"Failed to download file: {e}")

    def download_folder(self, prefix, download_path):
        """Downloads all files with a specific prefix from COS into a local folder."""
        try:
            objects = self.cos_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
            for obj in objects.get('Contents', []):
                object_name = obj['Key']
                file_path = os.path.join(download_path, os.path.relpath(object_name, prefix))

                # Create directories if they don't exist
                os.makedirs(os.path.dirname(file_path), exist_ok=True)

                # Download each file
                self.download_file(object_name, file_path)
        except Exception as e:
            print(f"Failed to download folder: {e}")

    def list_buckets(self):
        """Lists all available buckets in IBM Cloud Object Storage."""
        try:
            response = self.cos_client.list_buckets()
            bucket_names = [bucket['Name'] for bucket in response.get('Buckets', [])]
            print("Available buckets:", bucket_names)
            return bucket_names
        except Exception as e:
            print(f"Failed to list buckets: {e}")
            return []

    def delete_file(self, object_name):
        """Deletes a single file from COS."""
        try:
            self.cos_client.delete_object(Bucket=self.bucket_name, Key=object_name)
            print(f"File '{object_name}' deleted.")
        except Exception as e:
            print(f"Failed to delete file: {e}")

    def delete_folder(self, prefix):
        """Deletes all files with a specific prefix from COS, effectively removing a folder."""
        try:
            objects = self.cos_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
            delete_objects = [{'Key': obj['Key']} for obj in objects.get('Contents', [])]

            if delete_objects:
                self.cos_client.delete_objects(Bucket=self.bucket_name, Delete={'Objects': delete_objects})
                print(f"Folder '{prefix}' and all its contents deleted.")
            else:
                print(f"No objects found with prefix '{prefix}'.")
        except Exception as e:
            print(f"Failed to delete folder: {e}")


