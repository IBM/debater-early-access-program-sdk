import json

from WatsonStudioJobsManager import WatsonStudioJobsManager, CosFilesManager


# RUNTIME_ENV_CPU = "NLP + DO Runtime 24.1 on Python 3.11 XS"
# RUNTIME_ENV_GPU = "GPU V100 Runtime 24.1 on Python 3.11"


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
    api_key = "<api_key>"
    project_id = "<project_id>"
    job_name = "<job_name>"
    job_name = add_datetime(job_name)
    notebook_asset_ref = "<notebook asset ref>"  # can be found in the url when opening the notebook

    cos_api_key = '<cos_api_key>'
    cos_service_instance_id = '<cos_service_instance_id>'
    cos_endpoint_url = '<cos_endpoint_url>'
    cos_bucket_name = '<cos_bucket_name>'

    cos_files_manager = CosFilesManager(api_key=cos_api_key,
                                        service_instance_id=cos_service_instance_id,
                                        endpoint_url=cos_endpoint_url,
                                        bucket_name=cos_bucket_name)

    file_to_upload = '<csv_file>'
    cos_files_manager.upload_file(file_to_upload, '<csv_file_name>')

    job_input = {
        'domains_to_delete_before': ['test_domain'],
        'csvs_to_upload_data': [{'domain': 'test_domain',
                            # 'csvs_to_upload_folders': [{'folder_name': '', 'ids_column': '', 'texts_column': ''}],
                            'csvs_to_upload_files': [{'file_name': '<csv_file_name>', 'ids_column': '<ids_column>', 'texts_column': '<texts_column>'}]}],
        'kps_jobs_to_run': [{'domain': 'test_domain',
                             'results_folder_name': 'test_domain_result',
                             'delete_domain_when_finished': True
                             }],
        'domains_to_delete_after': [],
    }

    # cos_files_manager.upload_folder(folder_path='/Users/yoavkantor/Downloads/test_dir', prefix='test_dir')
    # cos_files_manager.download_folder(prefix='test_dir', download_path='/Users/yoavkantor/Downloads/test_dir2')

    jobs_manager = WatsonStudioJobsManager(api_key, project_id)
    # runtime_environment_asset_id = get_env_asset_id(manager, runtime_env_name)

    import base64
    job_input_str = json.dumps(job_input)
    job_input_b64 = base64.b64encode(job_input_str.encode('utf-8')).decode('ascii')
    env_variables = [f"kps_job_input={job_input_b64}"]
    job_id = jobs_manager.create_job(job_name, notebook_asset_ref, env_variables)
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

        for kps_jobs in job_input['kps_jobs_to_run']:
            results_folder_name = kps_jobs['results_folder_name']
            cos_files_manager.download_folder(results_folder_name, f'./kps_results/{results_folder_name}')

        # deletion_successful = jobs_manager.delete_job(job_id)
        # print("Job deleted:", deletion_successful)
    else:
        raise Exception(f'Job {job_id} failed')
