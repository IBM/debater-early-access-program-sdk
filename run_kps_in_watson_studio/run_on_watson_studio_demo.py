import dotenv
dotenv.load_dotenv('./.env')

import os
import json
from WatsonStudioJobsManager import WatsonStudioJobsManager, CosFilesManager


def get_env_asset_id(manager, runtime_env_name):
    res = manager.get_all_envs()
    asset_ids = [e['metadata']['asset_id'] for e in res if
     e['entity']['environment']['display_name'] == runtime_env_name]
    if len(asset_ids) != 1:
        raise Exception(f'unable to translate environment_name: {runtime_env_name} to one asset_id. got: {asset_ids}')
    return asset_ids[0]


def add_datetime(job_name):
    import datetime
    now = datetime.datetime.now()
    datetime_str = now.strftime('%Y_%m_%d__%H_%M_%S')
    return f"{job_name}_{datetime_str}"


if __name__ == '__main__':
    ws_api_key = os.getenv("WATSON_STUDIO_API_KEY")
    ws_project_id = os.getenv("WATSON_STUDIO_PROJECT_ID")
    ws_notebook_asset_ref = os.getenv("WATSON_STUDIO_NOTEBOOK_ASSET_REF")
    ws_job_name = add_datetime("kps_on_csv")

    cos_api_key = os.getenv("COS_API_KEY")
    cos_service_instance_id = os.getenv("COS_SERVICE_INSTANCE_ID")
    cos_endpoint_url = os.getenv("COS_ENDPOINT_URL")
    cos_bucket_name = os.getenv("COS_BUCKET_NAME")

    cos_files_manager = CosFilesManager(api_key=cos_api_key,
                                        service_instance_id=cos_service_instance_id,
                                        endpoint_url=cos_endpoint_url,
                                        bucket_name=cos_bucket_name)

    mongodb_use_remote_db = False  # When set to False - uses a non-persistent in-memory MongoDB
    mongodb_username = os.getenv("MONGODB_USERNAME")
    mongodb_password = os.getenv("MONGODB_PASSWORD")
    mongodb_endpoint = os.getenv("MONGODB_ENDPOINT")
    mongodb_certificate_locally = os.getenv("MONGODB_CERTIFICATE_LOCALLY")
    mongodb_certificate_in_cos = os.path.basename(mongodb_certificate_locally)
    cos_files_manager.upload_file(mongodb_certificate_locally, mongodb_certificate_in_cos)

    csv_file_locally = '/Users/yoavkantor/Library/CloudStorage/Box-Box/RAG Activities/agentic/ask_sales/feedbacks/datasets/meta_llama_llama_3_1_70b_instruct/ask_sales_feedbacks_kps_comments.csv'
    csv_file_in_cos = os.path.basename(csv_file_locally)
    cos_files_manager.upload_file(csv_file_locally, csv_file_in_cos)

    job_input = {
        'mongodb_params': {
            'mongodb_use_remote_db': mongodb_use_remote_db,
            # 'mongodb_username': mongodb_username,
            # 'mongodb_password': mongodb_password,
            # 'mongodb_endpoint': mongodb_endpoint,
            # 'mongodb_certificate_in_cos': mongodb_certificate_in_cos
        },
        'domains_to_delete_before': ['test_domain'],
        'create_domains': [{'domain': 'test_domain', 'domain_params': {}}],
        'csvs_to_upload_data': [{'domain': 'test_domain',
                            # 'csvs_to_upload_folders': [{'folder_name': '', 'ids_column': '', 'texts_column': ''}],
                            'csvs_to_upload_files': [{'file_name': csv_file_in_cos, 'ids_column': 'id', 'texts_column': 'comment'}]}],
        'kps_jobs_to_run': [{'domain': 'test_domain',
                             'results_folder_name': 'test_domain_result',
                             'delete_domain_when_finished': True,
                             'run_both_stances': False
                             }],
        'domains_to_delete_after': [],
    }

    # we pass job_input as json via COS and not in env-var since it's too big
    with open('./job_input.json', 'w') as json_file:
        json.dump(job_input, json_file, indent=4)
    cos_files_manager.upload_file('./job_input.json', 'job_input.json')

    cos_params = {
        'cos_api_key': cos_api_key,
        'cos_service_instance_id': cos_service_instance_id,
        'cos_endpoint_url': cos_endpoint_url,
        'cos_bucket_name': cos_bucket_name,
        'job_input_json': 'job_input.json'
    }

    import base64
    cos_params_str = json.dumps(cos_params)
    cos_params_b64 = base64.urlsafe_b64encode(cos_params_str.encode('utf-8')).decode('ascii')
    cos_params_b64 = cos_params_b64.rstrip('=')  # removing padding since it harms the var-env format
    env_variables = [f"cos_params={cos_params_b64}"]

    jobs_manager = WatsonStudioJobsManager(ws_api_key, ws_project_id)
    job_id = jobs_manager.create_job(ws_job_name, ws_notebook_asset_ref, env_variables)

    # job_id = manager.get_job_id_by_name(job_name)

    run_id = jobs_manager.start_run(job_id=job_id)
    # run_id = manager.get_last_run_id(job_id=job_id)

    # manager.cancel_run(job_id=job_id, run_id=run_id)
    # manager.delete_run(job_id=job_id, run_id=run_id)

    final_state = jobs_manager.wait_for_run(job_id=job_id, run_id=run_id)
    print("Final state:", final_state)

    if final_state == "completed":
        log_lines = jobs_manager.get_run_logs(job_id=job_id, run_id=run_id)
        log = "\n".join(log_lines)
        print(f'log: {log}')

        for kps_jobs in job_input.get('kps_jobs_to_run', []):
            results_folder_name = kps_jobs['results_folder_name']
            cos_files_manager.download_folder(results_folder_name, f'./kps_results/{results_folder_name}')

        # deletion_successful = jobs_manager.delete_job(job_id)
        # print("Job deleted:", deletion_successful)
    else:
        raise Exception(f'Job {job_id} failed')
