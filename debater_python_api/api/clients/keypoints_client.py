import json
import logging
import time
import calendar
import traceback
from enum import Enum

import pandas as pd
import requests

from debater_python_api.api.clients.key_point_summarization.KpsResult import KpsResult
from debater_python_api.api.clients.key_point_summarization.utils import get_default_request_header, \
    update_row_with_stance_data, validate_api_key_or_throw_exception, print_progress_bar, is_list_of_strings

from typing import List, Optional, Dict, Union, Any
from debater_python_api.api.clients.key_point_summarization.KpsExceptions import KpsIllegalInputException, \
    KpsNoPrivilegesException, KpsInvalidOperationException

domains_endpoint = '/domains'
comments_endpoint = '/comments'
kp_extraction_endpoint = '/kp_extraction'
data_endpoint = '/data'
report_endpoint = '/report'
comments_limit_endpoint = '/comments_limit'
self_check_endpoint = '/self_check'

class Stance(Enum):
    PRO = "pro"
    CON = "con"
    NO_STANCE = "no-stance"
    EACH_STANCE = "each-stance"

class KpsClient():
    '''
    A client for the Key Point Summarization (KPS) service.
    '''
    def __init__(self, apikey: str, host: Optional[str] = None, verify_certificate: bool = True, timeout_secs: int = 900):
        '''
        :param apikey: User's api-key, should be retreived from the early-access-program site.
        :param host: Optional, enable switching to alternative services.
        :param verify_certificate: Optional, will not verify the server's certificate when set to False. Useful when using a self-signed certificate.
        '''
        validate_api_key_or_throw_exception(apikey)
        self.apikey = apikey
        self.show_process = True
        self.host = host if host is not None else 'https://keypoint-matching-backend.debater.res.ibm.com'
        self.verify_certificate = verify_certificate
        self.api_version = "2"
        self.timeout = timeout_secs

    def _delete(self, url, params, retries=10, headers=None):
        return self._run_request_with_retry(requests.delete, url, params, retries, headers)

    def _get(self, url, params, retries=10, headers=None):
        return self._run_request_with_retry(requests.get, url, params, retries, headers)

    def _post(self, url, body, retries=10, headers=None):
        return self._run_request_with_retry(requests.post, url, body, retries, headers)

    def _run_request_with_retry(self, func, url, params, retries=10, headers_input=None):
        headers = get_default_request_header(self.apikey)
        if headers_input is not None:
            headers.update(headers_input)

        KpsClient._validate_request(params)
        params["api_version"] = self.api_version
        logging.info('client calls service (%s): %s' % (func.__name__, url))
        while True:
            try:
                if func.__name__ == 'post':
                    resp = func(url, json=params, headers=headers, timeout=self.timeout, verify=self.verify_certificate)
                else:
                    resp = func(url, params=params, headers=headers, timeout=self.timeout, verify=self.verify_certificate)
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code == 422:
                    msg = 'There is a problem with the request (%d): %s' % (resp.status_code, resp.reason)
                    logging.error(msg)
                    raise KpsIllegalInputException(msg)
                if resp.status_code == 403:
                    msg = 'User is not authorized to perform the requested operation (%d): %s' % (resp.status_code, resp.reason)
                    logging.error(msg)
                    raise KpsNoPrivilegesException(msg)
                msg = 'Failed calling server at %s: (%d) %s' % (url, resp.status_code, resp.reason)
            except (KpsIllegalInputException, KpsNoPrivilegesException) as e:
                raise e
            except Exception as e:
                track = traceback.format_exc()
                msg = "Can't access server at {}. Exception: {}".format(url, track)

            retries -= 1
            if retries < 0:
                raise ConnectionError(msg)
            else:
                logging.warning('%s, retries left: %d' % (msg, retries))
                time.sleep(10)

    def create_domain(self, domain:str, domain_params:Optional[Dict[str, Union[str, int, bool]]] = None, ignore_exists:Optional[bool] = False) -> None:
        """
        Create a new domain and customize domain's parameters.
        :param domain: the name of the new domain (must not exist already)
        :param domain_params: optional, a dictionary with various parameters for the domain.
        For full documentation of the domain
        params see https://github.com/IBM/debater-eap-tutorial/blob/main/survey_usecase/kps_parameters.pdf
        :param ignore_exists: optional, default False. When the domain already exists - if True, keeps existing domain.
        If False, raises an exception.
        """
        try:
            body = {'domain': domain}
            if domain_params is not None:
                KpsClient._validate_params_dict(domain_params, "domain")
                body['domain_params'] = domain_params
            self._post(url=self.host + domains_endpoint, body=body)
            logging.info('created domain: %s with domain_params: %s' % (domain, str(domain_params)))
        except KpsIllegalInputException as e:
            if 'already exist' not in str(e):
                raise e
            if not ignore_exists:
                raise KpsIllegalInputException(f'domain: {domain} already exists. To run on a new domain, please either select another '
                             f'domain name or delete this domain first by running client.delete_domain_cannot_be_undone({domain}).'
                             f'To allow running on an existing domain set ignore_exists=True ')

            logging.info(f'domain: {domain} already exists, domain_params are NOT updated.')

    def upload_comments(self, domain: str, comments_ids: List[str], comments_texts: List[str]) -> None:
        """
        Uploads comments into a domain. It is mandatory to create a domain before uploading comments into it.
        Re-uploading the same comments (same domain + comment_id + text) is relatively quick.
        Uploading an the same comment_id with a different text will raise an exception.
        Processing comments (cleaning + sentence splitting + calculating scores) takes some time. The progress will
        be displayed on screen and the method will return only when all comments are processed.
        :param domain: the name of the domain to upload the comments into. (usually one  per data-set).
        :param comments_ids: a list of comment ids (strings), comment ids must be unique, and composed of alphanumeric characters,
        spaces and underscores only.
        :param comments_texts: a list of comments (strings), this list must be the same length as comments_ids and the comment_id and comment_text should match by position in the list.
        """
        self.upload_comments_async(domain, comments_ids, comments_texts)
        logging.info('waiting for the comments to be processed')
        self.wait_till_all_comments_are_processed(domain)

    def upload_comments_async(self, domain: str, comments_ids: List[str], comments_texts: List[str]) -> None:
        '''
        Uploads comments into a domain in an async manner.
        It is mandatory to create a domain before uploading comments into it.
        Re-uploading the same comments (same domain + comment_id + text) is relatively quick.
        Uploading an the same comment_id with a different text will raise an exception.
        Processing comments (cleaning + sentence splitting + calculating scores) takes some time,
        please wait for it to finish before starting a key point summarization job, using get_comments_status or are_all_comments_processed.
        :param domain: the name of the domain to upload the comments into. (usually one  per data-set).
        :param comments_ids: a list of comment ids (strings), comment ids must be unique.
        :param comments_texts: a list of comments (strings), this list must be the same length as comments_ids and the comment_id and comment_text should match by position in the list.
        '''
        assert len(comments_ids) == len(comments_texts), 'comments_texts and comments_ids must be the same length'
        assert len(comments_ids) == len(set(comments_ids)), 'comments_ids must be unique'
        assert is_list_of_strings(comments_texts), 'comment_texts must be a list of strings'
        assert is_list_of_strings(comments_ids), 'comments_ids must be a list of strings'
        assert len([c for c in comments_texts if c is None or c == '' or len(c) == 0 or c.isspace()]) == 0, 'comment_texts must not have an empty string in it'
        assert len([c for c in comments_texts if len(c)>3000]) == 0, 'comment_texts must be shorter than 3000 characters'
        logging.info('uploading %d comments in batches' % len(comments_ids))

        ids_texts = list(zip(comments_ids, comments_texts))
        ids_texts.sort(key=lambda t: len(t[1]))
        batch_size = 2000
        uploaded = 0
        batches = [ids_texts[i:i + batch_size] for i in range(0, len(ids_texts), batch_size)]
        for batch in batches:
            comments_ids = [t[0] for t in batch]
            comments_texts = [t[1] for t in batch]
            body = {'domain': domain,
                    'comments_ids': comments_ids,
                    'comments_texts': comments_texts}

            self._post(url=self.host + comments_endpoint, body=body)
            uploaded += len(batch)
            logging.info('uploaded %d comments, out of %d' % (uploaded, len(ids_texts)))

    def get_comments_status(self, domain: str) -> Dict[str, int]:
        '''
        Get the status of the comments in a domain.
        All comments are processed and a kps job can start when there are no more pending comments.
        :param domain: the name of the domain
        :return: a dictionary with the status:
        * processed_comments: number of comments that where already processed
        * pending_comments: number of comments that still need to be processed
        * processed_sentences: number of sentences after sentence-splitting the processed comments
        '''
        res = self._get(self.host + comments_endpoint, {'domain': domain})
        logging.info('domain: %s, comments status: %s' % (domain, str(res)))
        return res

    def are_all_comments_processed(self, domain: str) -> bool:
        """
        Check if all comments in the domain are processed.
        :param domain: the name of the domain
        :return: True if all comments uploaded to the domain are processed.
        """
        res = self.get_comments_status(domain)
        return res['pending_comments'] == 0

    def wait_till_all_comments_are_processed(self, domain: str, polling_timeout_secs: Optional[int] = None) -> None:
        '''
        Waits for all comments in a domain to be processed.
        :param domain: the name of the domain
        :param polling_timeout_secs: optional, polling time in seconds
        '''
        while True:
            res = self.get_comments_status(domain)
            if res['pending_comments'] == 0:
                break
            time.sleep(polling_timeout_secs if polling_timeout_secs is not None else 10)

    @staticmethod
    def _get_run_params_with_stance(run_params, stance):
        if stance == Stance.NO_STANCE.value or not stance:
            return run_params
        if not run_params:
            run_params = {}
        else:
            assert "stances_to_run" not in run_params, "run_param 'stances_to_run' is no longer supported, please use the 'stance' method parameter instead."
            assert "stance" not in run_params, "run_param 'stance' is set interanlly by the client, please use the 'stance' method parameter instead."
            run_params = run_params.copy()

        if stance == Stance.PRO.value:
            run_params['stance'] = "PRO"
        elif stance == Stance.CON.value:
            run_params['stance'] = "CON"
        else:
            raise KpsIllegalInputException(f"Unsupported stance: *{stance}*, supported: no-stance, pro, con.")
        return run_params

    def run_full_kps_flow(self, domain: str, comments_texts: List[str],
                          stance: Optional[str] = Stance.EACH_STANCE.value):
        '''
        This is the simplest way to use the Key Point Summarization system.
        This method uploads the comments into a temporary domain, waits for them to be processed,
        starts a Key Point Summarization job using all comments, and waits for the results. Eventually, the domain is deleted.
        It is possible to use this method for up to 10000 comments. For longer jobs, please run the system in a staged
        manner (upload the comments yourself, start a job etc').
        If execution stopped before this method returned, please run client.delete_domain_cannot_be_undone(<domain>)
        to free resources and avoid longer waiting in future calls.  TODO Remove comment?
        :param domain: name of the temporary domain to store the comments. must not be a name of an existing domain, or
        an error will be raised. The domain must be composed of alphanumeric characters, spaces and underscores only.
        :param comments_texts: a list of comments (strings).
        :param stance: Optional, If "no-stance" - run on all the data disregarding the stance.
        If "pro", run on positive sentences only, if "con", run on con sentences (negative and suggestions).
        If "each-stance", starts two kps jobs, one for each stance, and returns the merged result object.
        :return: a KpsResult object with the result
        '''
        if len(comments_texts) > 10000:
            raise Exception(
                'Please use the stagged mode (upload_comments, run_kps_job) for jobs with more then 10000 comments')
        try:
            self.create_domain(domain, ignore_exists=False)
            comments_ids = [str(i) for i in range(len(comments_texts))]
            self.upload_comments(domain, comments_ids, comments_texts)
            if stance == Stance.EACH_STANCE.value:
                keypoint_matching = self.run_kps_job_both_stances(domain)
            else:
                keypoint_matching = self.run_kps_job(domain, stance=stance)
            return keypoint_matching
        except KpsIllegalInputException as e:
            if 'already exist' in str(e):
                raise KpsIllegalInputException(f'Domain {domain} already exists. Please use a new domain name or delete the domain using '
                                               f'delete_domain_cannot_be_undone() to run the full flow. If you wish to run kps on the existing domain, please'
                                               f'use the methods run_kps_job or run_kps_job_both_stances')
            else:
                raise e
        finally:
            self.delete_domain_cannot_be_undone(domain)

    def run_kps_job_both_stances(self, domain: str, comments_ids: Optional[List[str]]=None,
                                 run_params_pro:Optional[Dict[str, Any]] = None,
                                 run_params_con:Optional[Dict[str, Any]] = None,
                                 description: Optional[str] = None):
        """
        Starts two kps jobs simultaneously, one for pro sentences and one for con sentences, and returns a merge KpsResult.
        :param domain: the name of the domain
        :param comments_ids: optional, when None is passed, it uses all comments in the domain (typical usage) otherwise it only uses the comments according to the provided list of comments_ids.
        :param run_params_pro: optional,a dictionary with different parameters and their values, to be sent to the pro kps job.
        :param run_params_con: optional,a dictionary with different parameters and their values, to be sent to the con kps job.
        For full documentation of supported run_params see https://github.com/IBM/debater-eap-tutorial/blob/main/survey_usecase/kpa_parameters.pdf
        :param description: optional, add a textual description to a job so it will be easy to detect it in the user-report. for each job, the stance will be appended to the description.,
        the stance of each job will be added to the description
        :return: a KpsResult object with the merged pro and con result.
        """
        description_pro = (description + " (pro)") if description else None
        description_con = (description + " (con)") if description else None
        future_pro = self.run_kps_job_async(domain, comments_ids, stance=Stance.PRO.value, run_params=run_params_pro, description = description_pro)
        future_con = self.run_kps_job_async(domain, comments_ids, stance=Stance.CON.value, run_params=run_params_con, description = description_con)
        result_pro = future_pro.get_result()
        result_con = future_con.get_result()
        return KpsResult.get_merged_pro_con_results(pro_result=result_pro, con_result=result_con)

    def run_kps_job(self, domain: str, comments_ids: Optional[List[str]]=None,
                    run_params: Optional[Dict[str, Any]] = None,
                    description: Optional[str] = None, stance: Optional[str]=Stance.NO_STANCE.value) -> 'KpsResult':
        """
        Runs Key Point Summarization (KPS) in a synchronous manner: starts the job, waits for the results and return them.
        Please make sure all comments had already been uploaded into a domain and processed before starting a new job (using the get_comments_status or are_all_comments_processed methods).
        :param domain: the name of the domain
        :param comments_ids: optional, when None is passed, it uses all comments in the domain (typical usage) otherwise it only uses the comments according to the provided list of comments_ids.
        :param run_params: optional,a dictionary with different parameters and their values. For full documentation of supported run_params see https://github.com/IBM/debater-eap-tutorial/blob/main/survey_usecase/kpa_parameters.pdf
        :param description: optional, add a textual description to a job so it will be easy to detect it in the user-report.
        :param stance: Optional, default to "no-stance". If "no-stance" - run on all the data disregarding the stance.
        If "pro", run on positive sentences only, if "con", run on con sentences (negative and suggestions).
        :return: a KpsResult object with the result.
        """
        future = self.run_kps_job_async(domain, run_params=run_params, comments_ids=comments_ids, stance=stance, description=description)
        keypoint_matching = future.get_result(high_verbosity=True)
        return keypoint_matching

    def run_kps_job_async(self, domain: str, comments_ids: Optional[List[str]]=None,
                          stance: Optional[str] = Stance.NO_STANCE.value,
                          run_params: Optional[Dict[str, Any]] = None, description: Optional[str] = None) -> 'KpsJobFuture':
        """
        Starts a Key Point Summarization (KPS) job in an async manner. Please make sure all comments had already been
        uploaded into a domain and processed before starting a new job (using the using get_comments_status or are_all_comments_processed methods).
        :param domain: the name of the domain
        :param comments_ids: optional, when None is passed, it uses all comments in the domain (typical usage) otherwise it only uses the comments according to the provided list of comments_ids.
        :param run_params: optional, a dictionary with different parameters and their values. For full documentation of supported run_params see https://github.com/IBM/debater-eap-tutorial/blob/main/survey_usecase/kpa_parameters.pdf
        :param stance: Optional, default to "no-stance". If "no-stance" - run on all the data disregarding the stance.
        If "pro", run on positive sentences only, if "con", run on con sentences (negative and suggestions).
        :param description: optional, add a textual description to a job so it will be easy to detect it in the user-report.
        :return: KpsJobFuture: an object that enables the retrieval of the results in an async manner
        """
        if not self.are_all_comments_processed(domain):
            raise KpsInvalidOperationException(f"Attempt to start a KPS job on domain {domain} when comments"
                                               f" are still processing. Please wait until all comments are processed"
                                               f" before starting a kps job. You can test the comments status using the"
                                               f"methods are_all_comments_processed({domain}) or get_comments_status({domain})")
        run_params = KpsClient._get_run_params_with_stance(run_params, stance)
        body = {'domain': domain}

        if comments_ids is not None:
            assert is_list_of_strings(comments_ids), 'comments_ids must be a list of strings'
            body['comments_ids'] = comments_ids

        if run_params is not None:
            KpsClient._validate_params_dict(run_params, 'run')
            body['run_params'] = run_params

        if description is not None:
            body['description'] = description

        res = self._post(url=self.host + kp_extraction_endpoint, body=body)
        logging.info(f'started a kp summarization job - domain: {domain}, stance: {stance}, run_params: {run_params}, {"" if description is None else f"description: {description}, "}job_id: {res["job_id"]}')
        return KpsJobFuture(self, res['job_id'])

    def get_kps_job_status(self, job_id: str):
        '''
        Checks for the status of a key point summarization job. It returns a json with a 'status' key that can have one of the following values: PENDING, PROCESSING, DONE, CANCELED, ERROR
        If the status is PROCESSING, it also has a 'progress' key that describes the calculation progress.
        If the status is DONE, it also has a 'result' key that has the result_json, that can be converted into KpsResults using KpsResult.create_from_result_json(result_json)
        If the status is ERROR, it also has a 'error_msg' key that has the description of the error.
        :param job_id: the job_id (can be found in the future returned when the job was started or in the user-report)
        :return: see description above.
        '''
        params = {'job_id': job_id}
        return self._get(self.host + kp_extraction_endpoint, params)

    def cancel_kps_job(self, job_id: str):
        '''
        Stops a running key point summarization job.
        :param job_id: the job_id
        :return: the request's response
        '''
        logging.info(f"Canceling job {job_id}")
        try:
            return self._delete(self.host + kp_extraction_endpoint, {'job_id': job_id})
        except KpsIllegalInputException as e:
            return

    def cancel_all_kps_jobs_for_domain(self, domain: str):
        '''
        Stops all running jobs and cancels all pending jobs in a domain.
        :param domain: the name of the domain.
        :return: the request's response
        '''
        logging.info(f"Canceling all jobs for domain {domain}")
        return self._delete(self.host + data_endpoint, {'domain': domain, 'clear_kp_analysis_jobs_log': False, 'clear_db': False})

    def cancel_all_kps_jobs_all_domains(self):
        '''
        Stops all running jobs and cancels all pending jobs in all domains.
        :return: the request's response
        '''
        logging.info(f"Canceling all jobs for all domains.")
        return self._delete(self.host + data_endpoint, {'clear_kp_analysis_jobs_log': False, 'clear_db': False})

    def delete_domain_cannot_be_undone(self, domain: str):
        '''
        Deletes a domain. Stops all running jobs and cancels all pending jobs in a domain. Erases the data (comments and sentences) in a domain and clears the domain's cache.
        When uploaded comments in a domain need to be replaced, first delete the domain and then upload the updated comments.
        :param domain: the name of the domain
        :return: the request's response
        '''
        try:
            resp = self._delete(self.host + data_endpoint, {'domain': domain, 'clear_kp_analysis_jobs_log': False, 'clear_db': True})
            logging.info(f'domain: {domain} was deleted')
            return resp
        except KpsIllegalInputException as e:
            if 'doesn\'t have domain' not in str(e):
                raise e
            logging.info(f'domain: {domain} doesn\'t exist.')

    def delete_all_domains_cannot_be_undone(self):
        '''
        Deletes all user's domains. Stops all running jobs and cancels all pending jobs in all domains. Erases the data (comments and sentences) in all domains and clears all domains' caches.
        :return: the request's response
        '''
        return self._delete(self.host + data_endpoint, {'clear_kp_analysis_jobs_log': False, 'clear_db': True})

    def get_full_report(self, days_ago=30, active_domains_only=False):
        '''
        Retreives a json with the user's report.
        :param days_ago: key point summarization jobs older then this parameter will be filtered out
        :param active_domains_only: If true, returns only jobs of domains that were not deleted
        returns: The report which consists:
          * 'comments_status': all the domains that the user have and the current status of each domain (number of processed comments, sentences and comments that still need to be processed, similar to get_comments_status method).
          * 'kp_summarization_status': a list of all key point summarization jobs that the user have/had with all the relevant details and parameters for each job.
        '''
        user_report = self._get(self.host + report_endpoint, {'days_ago': days_ago})

        if active_domains_only:
            domains = [domain_status['domain'] for domain_status in user_report["domains_status"]]
            user_report['kp_analysis_status'] = list(filter(lambda x: x["domain"] in domains, user_report['kp_analysis_status']))
        return user_report

    def get_comments_limit(self) -> int:
        '''
        Retreives a json with the permitted number of comments per KPS job.
        returns:
          * 'n_comments_limit': The maximal number of comments permitted per KPS job (None if there is no limit).
        '''
        return self._get(self.host + comments_limit_endpoint, {})

    def run_self_check(self):
        '''
        Checks the connection to the service and if the service is UP and running.
        :return: a json with 'status': that have the value UP if all is well and DOWN otherwise.
        '''
        return self._get(self.host + self_check_endpoint, {})

    def get_unmapped_sentences_for_kps_result(self, kps_result: KpsResult):
        '''
        Retrieve all unmapped sentences associated with the kps_result.
        :param kps_result: KpsResult object storing the results.
        :return: a dataframe with all unmapped sentences and their data, or None if domain
        '''
        domain = kps_result.get_domain()
        job_ids = list(kps_result.get_stance_to_job_id().values())
        sentences_dfs = []

        if len(job_ids) == 1:
            sents_df = self.get_sentences_for_domain(domain, job_ids[0])
        elif len(job_ids) == 2:
            for job_id in job_ids:
                sents_df = self.get_sentences_for_domain(domain, job_id)
                sentences_dfs.append(sents_df)
            sents_df = pd.concat(sentences_dfs).drop_duplicates(subset=["comment_id","sent_id"])

        if len(sents_df) == 0:
            logging.info("No sentences found for result, maybe the domain was deleted?")
            return None

        mapped_sents_df = kps_result.result_df.rename(columns={"sentence_id":"sent_id"})
        unmapped_sents_df = pd.concat([sents_df,mapped_sents_df]).drop_duplicates(subset=["sent_id","comment_id"], keep=False)
        unmapped_sents_df = unmapped_sents_df[sents_df.columns]
        return unmapped_sents_df

    def get_sentences_for_domain(self, domain: str, job_id: Optional[str] = None):
        '''
        Uploaded comments are cleaned and split into sentences. This method retrieves the sentences in a domain.
        :param domain: the name of the domain.
        :param job_id: when provided, it will only return the sentences used in a specific key point summarization job.
        :return: a dataframe with all the sentences' data.
        '''
        params = {'domain': domain, 'get_sentences': True}
        if job_id:
            params['job_id'] = job_id

        res = self._get(self.host + data_endpoint, params=params)
        logging.info(res['msg'])
        sentences = res['sentences_results']
        if len(sentences) == 0:
            logging.info(f"No sentences found, returning empty dataframe.")
            return pd.DataFrame()

        cols = list(sentences[0].keys())
        rows = [[s[col] for col in cols] for s in sentences]
        df = pd.DataFrame(rows, columns=cols)
        if "stance_dict" in cols and sentences[0]["stance_dict"]:
            df = df.apply(lambda r: update_row_with_stance_data(r), axis=1)
        return df

    @staticmethod
    def init_logger():
        '''
        Inits the logger for more informative console prints.
        '''
        from logging import getLogger, getLevelName, Formatter, StreamHandler
        log = getLogger()
        log.setLevel(getLevelName('INFO'))
        log_formatter = Formatter("%(asctime)s [%(levelname)s] %(filename)s %(lineno)d: %(message)s")

        console_handler = StreamHandler()
        console_handler.setFormatter(log_formatter)
        log.handlers = []
        log.addHandler(console_handler)

    @staticmethod
    def print_report(user_report, job_statuses: Optional[List[str]] = None, active_domains_only: Optional[bool] = True,
                     domains_to_show: Optional[List[str]] = None):
        '''
        Prints the user_report to console.
        :param user_report: the user report, returned by method get_full_report
        :param job_statuses: optional, print only jobs with the listed statuses. possible statuses:
        'PENDING','PROCESSING','CANCELED','ERROR','DONE'
        :param active_domains_only: optional, print only jobs from domains that were not deleted. default: true.
        :param domains_to_show: optional, print only these domain and jobs from these domains (if they exist).
        '''
        logging.info('User Report:')
        comments_statuses = user_report['domains_status']
        logging.info('  Comments status by domain (%d domains):' % len(comments_statuses))
        active_domains = []
        if len(comments_statuses) == 0:
            logging.info('    User has no domains')
        else:
            for status in comments_statuses:
                domain = status["domain"]
                if domains_to_show and domain not in domains_to_show:
                    continue
                active_domains.append(domain)
                logging.info(
                    f'    Domain: {domain}, Domain Params: {status["domain_params"]}, Status: {status["data_status"]}')
        kp_summarization_statuses = user_report['kp_analysis_status']
        logging.info(f'  Key point summarization - jobs status:')
        if len(kp_summarization_statuses) == 0:
            logging.info('    User has no key point summarization jobs history')
        else:
            n_total_jobs = len(kp_summarization_statuses)
            if active_domains_only:
                kp_summarization_statuses = list(filter(lambda x: x["domain"] in active_domains, kp_summarization_statuses))
            if job_statuses and len(job_statuses) > 0:
                kp_summarization_statuses = list(filter(lambda x: x["status"] in job_statuses, kp_summarization_statuses))
            if domains_to_show and len(domains_to_show) > 0:
                kp_summarization_statuses = list(filter(lambda x: x["domain"] in domains_to_show, kp_summarization_statuses))
            n_displayed_jobs = len(kp_summarization_statuses)
            logging.info(
                f'  Displaying {n_displayed_jobs} jobs out of {n_total_jobs}: {"only active" if active_domains_only else "all"} domains, '
                f'{"all" if not job_statuses else job_statuses} jobs statuses, {"all" if not domains_to_show else domains_to_show} domain names')

            for kp_summarization_status in kp_summarization_statuses:
                logging.info(f'    Job: {str(kp_summarization_status)}')

    def get_results_from_job_id(self, job_id:str):
        kps_future = KpsJobFuture(self, job_id)
        kps_result = kps_future.get_result(high_verbosity=True)
        return kps_result

    @staticmethod
    def _validate_params_dict(params_dict, params_type:str):
        if params_dict is None:
            return
        if not isinstance(params_dict, dict):
            raise KpsIllegalInputException(f'{params_type}_params must be a dictionary, given: {type(params_dict)}')

        for k,v in params_dict.items():
            if not isinstance(k, str):
                raise KpsIllegalInputException(f'{params_type}_params keys must be a strings, given: {type(k)}:{k}')
            if type(v) not in [list, str, int, float, bool]:
                raise KpsIllegalInputException(f'unsupported {params_type}_params value type: {type(v)}:{v}')

    @staticmethod
    def _validate_request(params):
        for param_name,val in params.items():
            if param_name in ["domain","job_id","description"]:
                assert isinstance(val, str), f"{param_name} should be a string, given {type(val)} : {val}"
            try:
                json.dumps(val)
            except TypeError as e:
                raise TypeError(f"Request could not be sent to server due to invalid parameters: {param_name}:{val}:\n{e}")



class KpsJobFuture:
    '''
    A future for an async key point summarization job. Wraps the job_id and uses a provided client for retrieving the job's result.
    Usually created when starting a key point summarization job but can also be created by a user, by suppling the client and the job_id.
    The job_id can be retrieved from the console (it is printed to console when a job is started) or from the user-report.
    '''
    def __init__(self, client: KpsClient, job_id: str):
        '''
        Create a KpsJobFuture over a job_id for results retrieval.
        :param client: a client for communicating with the service.
        :param job_id: the job_id. The job_id can be retrieved from the console (it is printed to console when a job is started) or from the user-report.
        '''
        self.client = client
        self.job_id = job_id
        self.polling_timeout_secs = 30

    def get_job_id(self) -> str:
        '''
        :return: the job_id
        '''
        return self.job_id

    def get_result(self,
                   dont_wait: bool = False, wait_secs: Optional[int] = None, polling_timeout_secs: Optional[int] = None,
                   high_verbosity: bool = True) -> KpsResult:
        '''
        Retreives the job's result. This method polls and waits till the job is done and the result is available.
        :param dont_wait: when True, tries to get the result once and returns it if it's available, otherwise returns None.
        :param wait_secs: limit the waiting time (in seconds).
        :param polling_timeout_secs: sets the time to wait before polling again (in seconds). The default is 30 seconds.
        :param high_verbosity: set to False to reduce the number of messages printed to the logger.
        :return: the KpsResult object or throws an exception if an error occurs or if the job was canceled.
        '''
        start_time = time.time()

        do_again = True
        while do_again:
            result = self.client.get_kps_job_status(self.job_id)
            if result['status'] == 'PENDING':
                if high_verbosity:
                    logging.info('job_id %s is pending' % self.job_id)
            elif result['status'] == 'PROCESSING':
                if high_verbosity:
                    progress = result['progress']
                    logging.info('job_id %s is running, progress: %s' % (self.job_id, progress))
                    self._print_progress_bar(progress)
            elif result['status'] == 'DONE':
                logging.info('job_id %s is done, returning result' % self.job_id)
                json_results = result['result']
                return KpsResult.create_from_result_json(json_results)
            elif result['status'] == 'ERROR':
                error_msg = 'job_id %s has error, error_msg: %s' % (self.job_id, str(result['error_msg']))
                logging.error(error_msg)
                raise Exception(error_msg)
            elif result['status'] == 'CANCELED':
                logging.info('job_id %s was canceled!' % self.job_id)
                raise Exception('waiting for result on a job that was canceled')
            else:
                raise Exception('unsupported status: %s, result: %s' % (result['status'], str(result)))

            do_again = False if dont_wait else True if wait_secs is None else time.time() - start_time < wait_secs
            time.sleep(polling_timeout_secs if polling_timeout_secs is not None else self.polling_timeout_secs)
        return None

    def cancel(self):
        '''
        Cancels (stops) the running job. Please stop unneeded jobs since they use a lot of resources.
        '''
        self.client.cancel_kps_job(self.job_id)

    def _print_progress_bar(self, progress):
        if 'total_stages' in progress:
            total_stages = progress['total_stages']
            for i in reversed(range(total_stages)):
                stage = str(i + 1)
                stage_i = 'stage_' + stage
                if stage_i in progress and 'inferred_batches' in progress[stage_i] and 'total_batches' in progress[stage_i]:
                    print_progress_bar(progress[stage_i]['inferred_batches'], progress[stage_i]['total_batches'], prefix='Stage %s/%s:' % (stage, str(total_stages)), suffix='Complete', length=50)
                    break
