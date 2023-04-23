import logging
import time
import calendar
import traceback

import requests
from debater_python_api.api.clients.abstract_client import AbstractClient
from debater_python_api.utils.general_utils import get_default_request_header
from typing import List, Optional, Dict
from debater_python_api.utils.kp_analysis_utils import print_progress_bar
from debater_python_api.api.clients.key_point_analysis.KpaExceptions import KpaIllegalInputException

domains_endpoint = '/domains'
comments_endpoint = '/comments'
kp_extraction_endpoint = '/kp_extraction'
data_endpoint = '/data'
report_endpoint = '/report'
comments_limit_endpoint = '/comments_limit'
self_check_endpoint = '/self_check'


class KpAnalysisClient(AbstractClient):
    '''
    A client for the Key Point Analysis (KPA) service.
    '''
    def __init__(self, apikey: str, host: Optional[str] = None, allow_self_signed_certificates: Optional[bool] = False):
        '''
        :param apikey: user's api-key, should be retreived from the early-access-program site.
        :param host: optional, enable switching to alternative services.
        :param allow_self_signed_certificates: optional, enable self-signed TLS certificate, default False
        '''
        AbstractClient.__init__(self, apikey)
        self.host = host if host is not None else 'https://keypoint-matching-backend.debater.res.ibm.com'
        self.allow_self_signed_certs = allow_self_signed_certificates

    def _delete(self, url, params, timeout=300, retries=10, headers=None):
        return self._run_request_with_retry(requests.delete, url, params, timeout, retries, headers)

    def _get(self, url, params, timeout=300, retries=10, headers=None):
        return self._run_request_with_retry(requests.get, url, params, timeout, retries, headers)

    def _post(self, url, body, timeout=300, retries=10, headers=None):
        return self._run_request_with_retry(requests.post, url, body, timeout, retries, headers)

    def _run_request_with_retry(self, func, url, params, timeout=20, retries=10, headers_input=None):
        headers = get_default_request_header(self.apikey)
        if headers_input is not None:
            headers.update(headers_input)

        logging.info('client calls service (%s): %s' % (func.__name__, url))
        while True:
            try:
                verify_cert = not self.allow_self_signed_certs
                if func.__name__ == 'post':
                    resp = func(url, json=params, headers=headers, timeout=timeout, verify=verify_cert)
                else:
                    resp = func(url, params=params, headers=headers, timeout=timeout, verify=verify_cert)
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code == 422:
                    msg = 'There is a problem with the request (%d): %s' % (resp.status_code, resp.reason)
                    logging.error(msg)
                    raise KpaIllegalInputException(msg)
                msg = 'Failed calling server at %s: (%d) %s' % (url, resp.status_code, resp.reason)
            except KpaIllegalInputException as e:
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

    def _is_list_of_strings(self, lst):
        return isinstance(lst, list) and len([a for a in lst if not isinstance(a, str)]) == 0

    def create_domain(self, domain, domain_params=None):
        """
        Create a new domain and customize domain's parameters.
        :param domain: the name of the new domain (must not exist already)
        :param domain_params: a dictionary with various parameters for the domain. For full documentation of the domain
        params see https://github.com/IBM/debater-eap-tutorial/blob/main/survey_usecase/kpa_parameters.pdf
        """
        body = {'domain': domain}
        if domain_params is not None:
            body['domain_params'] = domain_params

        self._post(url=self.host + domains_endpoint, body=body)
        logging.info('created domain: %s with domain_params: %s' % (domain, str(domain_params)))


    def upload_comments(self, domain: str, comments_ids: List[str], comments_texts: List[str], batch_size: int = 2000) -> None:
        '''
        Uploads comments into a domain. It is not mandatory to create a domain before uploading comments into it.
        If the domain doesn't exist, a domain with default parameters will be created.
        When we need to change domain's parameters, we must create it first via create_domain method.
        Re-uploading the same comments (same domain + comment_id, text is ignored) is not problematic (and relativly quick).
        Processing comments (cleaning + sentence splitting + calculating quality etc.) takes some time,
        please wait for it to finish before starting a key point analysis job (using method wait_till_all_comments_are_processed).
        :param domain: the name of the domain to upload the comments into. (usually one  per data-set).
        :param comments_ids: a list of comment ids (strings), comment ids must be unique.
        :param comments_texts: a list of comments (strings), this list must be the same length as comments_ids and the comment_id and comment_text should match by position in the list.
        :param batch_size: the number of comments that will be uploaded in every REST-API call.
        '''
        assert len(comments_ids) == len(comments_texts), 'comments_texts and comments_ids must be the same length'
        assert len(comments_ids) == len(set(comments_ids)), 'comment_ids must be unique'
        assert self._is_list_of_strings(comments_texts), 'comment_texts must be a list of strings'
        assert self._is_list_of_strings(comments_ids), 'comment_ids must be a list of strings'
        assert len([c for c in comments_texts if c is None or c == '' or len(c) == 0 or c.isspace()]) == 0, 'comment_texts must not have an empty string in it'
        logging.info('uploading %d comments in batches' % len(comments_ids))

        ids_texts = list(zip(comments_ids, comments_texts))
        uploaded = 0
        batches = [ids_texts[i:i + batch_size] for i in range(0, len(ids_texts), batch_size)]
        for batch in batches:
            comments_ids = [t[0] for t in batch]
            comments_texts = [t[1] for t in batch]
            body = {'domain': domain,
                    'comments_ids': comments_ids,
                    'comments_texts': comments_texts}

            self._post(url=self.host + comments_endpoint, body=body, retries=10)
            uploaded += len(batch)
            logging.info('uploaded %d comments, out of %d' % (uploaded, len(ids_texts)))

    def get_comments_status(self, domain: str) -> Dict[str, int]:
        '''
        Get the status of the comments in a domain.
        :param domain: the name of the domain
        :return: a dictionary with the status:
        * processed_comments: number of comments that where already processed
        * pending_comments: number of comments that still need to be processed
        * processed_sentences: number of sentences after sentence-splitting the processed comments
        '''
        res = self._get(self.host + comments_endpoint, {'domain': domain})
        logging.info('domain: %s, comments status: %s' % (domain, str(res)))
        return res

    def wait_till_all_comments_are_processed(self, domain: str, polling_timout_secs: Optional[int] = None) -> None:
        '''
        Waits for all comments in a domain to be processed.
        :param domain: the name of the domain
        '''
        while True:
            res = self.get_comments_status(domain)
            if res['pending_comments'] == 0:
                break
            time.sleep(polling_timout_secs if polling_timout_secs is not None else 10)

    def start_kp_analysis_job(self, domain: str, comments_ids: Optional[List[str]]=None,
                              run_params=None, description: Optional[str]=None) -> 'KpAnalysisTaskFuture':
        """
        Starts a Key Point Analysis (KPA) job in an async manner. Please make sure all comments had already been
        uploaded into a domain and processed before starting a new job (using the wait_till_all_comments_are_processed method).
        :param domain: the name of the domain
        :param comments_ids: when None is passed, it uses all comments in the domain (typical usage) otherwise it only uses the comments according to the provided list of comment_ids.
        :param run_params: a dictionary with different parameters and their values. For full documentation of supported run_params see https://github.com/IBM/debater-eap-tutorial/blob/main/survey_usecase/kpa_parameters.pdf
        :param description: add a description to a job so it will be easy to detect it in the user-report.
        :return: KpAnalysisTaskFuture: an object that enables the retrieval of the results in an async manner.
        """

        # TODO validate run_params
        body = {'domain': domain}

        if comments_ids is not None:
            body['comments_ids'] = comments_ids

        if run_params is not None:
            body['run_params'] = run_params

        if description is not None:
            body['description'] = description

        res = self._post(url=self.host + kp_extraction_endpoint, body=body)
        logging.info(f'started a kp analysis job - domain: {domain}, run_params: {run_params}, {"" if description is None else f"description: {description}, "}job_id: {res["job_id"]}')
        return KpAnalysisTaskFuture(self, res['job_id'])

    def get_kp_extraction_job_status(self, job_id: str, top_k_kps: Optional[int] = None,
                                     top_k_sentences_per_kp: Optional[int] = None):
        '''
        Checks for the status of a key point analysis job. It returns a json with a 'status' key that can have one of the following values: PENDING, PROCESSING, DONE, CANCELED, ERROR
        If the status is PROCESSING, it also have a 'progress' key that describes the calculation progress.
        If the status is DONE, it also have a 'result' key that has the result json.
        If the status is ERROR, it also have a 'error_msg' key that has the description of the error.
        The result json have the following structure:
            * 'keypoint_matchings': a list of keypoint_matching (key point and its matched sentences). Sorted descendingly according to number of matched sentences. each keypoint_matching have:
                * 'keypoint': the key point (string).
                * 'matching': a list of matches (sentences that match the key point). each match have the sentences details ('domain', 'comment_id', 'sentence_id', 'sents_in_comment', 'span_start', 'span_end', 'num_tokens', 'argument_quality', 'sentence_text') and a match score ('score') this is the match score between the sentence and the key point. The matchings are sorted descendingly according to their match score.
        :param job_id: the job_id (can be found in the future returned when the job was started or in the user-report)
        :param top_k_kps: use this parameter to truncate the result json to have only the top K key points.
        :param top_k_sentences_per_kp: use this parameter to truncate the result json to have only the top K matched sentences per key point.
        :return: see description above.
        '''
        params = {'job_id': job_id}

        if top_k_kps is not None:
            params['top_k_kps'] = top_k_kps

        if top_k_sentences_per_kp is not None:
            params['top_k_sentences_per_kp'] = top_k_sentences_per_kp

        return self._get(self.host + kp_extraction_endpoint, params, timeout=180)

    def run(self, comments_texts: List[str], comments_ids: Optional[List[str]]=None):
        '''
        This is the simplest way to use the Key Point Analysis system.
        This method uploads the comments into a temporary domain, waits for them to be processed, starts a Key Point Analysis job using all comments (auto key points extraction with default parameters), and waits for the results. Eventually, the domain is deleted.
        It is possible to use this method for up to 1000 comments. For longer jobs, please run the system in a stagged manner (upload the comments yourself, start a job etc').
        :param comments_texts: a list of comments (strings).
        :param comments_ids: (optional) a list of comment ids (a list of strings). When not provided, dummy comment_ids will be generated (1, 2, 3,...). When provided, comment_ids must be unique, must be the same length as comments_texts and the comment_id and comment_text should match by position in the list.
        :return: a json with the result
        '''
        if len(comments_texts) > 10000:
            raise Exception('Please use the stagged mode (upload_comments, start_kp_analysis_job) for jobs with more then 10000 comments')

        if comments_ids is None:
            comments_ids = [str(i) for i in range(len(comments_texts))]
        domain = 'run_temp_domain_' + str(calendar.timegm(time.gmtime()))
        logging.info('uploading comments')
        self.upload_comments(domain, comments_ids, comments_texts)
        logging.info('waiting for the comments to be processed')
        self.wait_till_all_comments_are_processed(domain)
        logging.info('starting the key point analysis job')
        future = self.start_kp_analysis_job(domain)
        logging.info('waiting for the key point analysis job to finish')
        keypoint_matching = future.get_result(high_verbosity=True)
        self.delete_domain_cannot_be_undone(domain)
        return keypoint_matching

    def cancel_kp_extraction_job(self, job_id: str):
        '''
        Stops a running key point analysis job.
        :param job_id: the job_id
        :return: the request's response
        '''
        return self._delete(self.host + kp_extraction_endpoint, {'job_id': job_id})

    def cancel_all_extraction_jobs_for_domain(self, domain: str):
        '''
        Stops all running jobs and cancels all pending jobs in a domain.
        :param domain: the name of the domain.
        :return: the request's response
        '''
        return self._delete(self.host + data_endpoint, {'domain': domain, 'clear_kp_analysis_jobs_log': False, 'clear_db': False})

    def cancel_all_extraction_jobs_all_domains(self):
        '''
        Stops all running jobs and cancels all pending jobs in all domains.
        :return: the request's response
        '''
        return self._delete(self.host + data_endpoint, {'clear_kp_analysis_jobs_log': False, 'clear_db': False})

    def delete_domain_cannot_be_undone(self, domain: str):
        '''
        Deletes a domain. Stops all running jobs and cancels all pending jobs in a domain. Erases the data (comments and sentences) in a domain and clears the domain's cache.
        When uploaded comments in a domain need to be replaced, first delete the domain and then upload the updated comments.
        :param domain: the name of the domain
        :return: the request's response
        '''
        return self._delete(self.host + data_endpoint, {'domain': domain, 'clear_kp_analysis_jobs_log': False, 'clear_db': True})

    def delete_all_domains_cannot_be_undone(self):
        '''
        Deletes all user's domains. Stops all running jobs and cancels all pending jobs in all domains. Erases the data (comments and sentences) in all domains and clears all domains' caches.
        :return: the request's response
        '''
        return self._delete(self.host + data_endpoint, {'clear_kp_analysis_jobs_log': False, 'clear_db': True})

    def get_full_report(self, days_ago=30):
        '''
        Retreives a json with the user's report.
        :param days_ago: key point analysis jobs older then this parameter will be filtered out
        returns: The report which consists:
          * 'comments_status': all the domains that the user have and the current status of each domain (number of processed comments, sentences and comments that still need to be processed, similar to get_comments_status method).
          * 'kp_analysis_status': a list of all key point analysis jobs that the user have/had with all the relevant details and parameters for each job.
        '''
        return self._get(self.host + report_endpoint, {'days_ago': days_ago}, timeout=180)

    def get_comments_limit(self):
        '''
        Retreives a json with the permitted number of comments per KPA job.
        returns:
          * 'n_comments_limit': The maximal number of comments permitted per KPA job (None if there is no limit).
        '''
        return self._get(self.host + comments_limit_endpoint, {}, timeout=180)

    def run_self_check(self):
        '''
        Checks the connection to the service and if the service is UP and running.
        :return: a json with 'status': that have the value UP if all is well and DOWN otherwise.
        '''
        return self._get(self.host + self_check_endpoint, None, timeout=180)

    def get_sentences_for_domain(self, domain: str, job_id: Optional[str] = None):
        '''
        Uploaded comments are cleaned and splitted into sentences. This method retrieves the sentences in a domain.
        :param domain: the name of the domain.
        :param job_id: when provided, it will only return the sentences used in a specific key point analysis job.
        :return: a dictionary with all the sentences' details.
        '''
        res = self._get(self.host + data_endpoint, {'domain': domain, 'get_sentences': True, 'job_id': job_id})
        logging.info(res['msg'])
        return res['sentences_results']

    def create_domain_ignore_exists(self, domain, domain_params):
        try:
            self.create_domain(domain, domain_params)
            logging.info(f'domain: {domain} was created')
        except KpaIllegalInputException as e:
            if 'already exist' not in str(e):
                raise e
            logging.info(f'domain: {domain} already exists, domain_params are NOT updated.')

    def delete_domain_ignore_doesnt_exist(self, domain):
        try:
            self.delete_domain_cannot_be_undone(domain)
            logging.info(f'domain: {domain} was deleted')
        except KpaIllegalInputException as e:
            if 'doesn\'t have domain' not in str(e):
                raise e
            logging.info(f'domain: {domain} doesn\'t exist.')


class KpAnalysisTaskFuture:
    '''
    A future for an async key point analysis job. Wraps the job_id and uses a provided client for retrieving the job's result.
    Usually created when starting a key point analysis job but can also be created by a user, by suppling the client and the job_id.
    The job_id can be retrieved from the console (it is printed to console when a job is started) or from the user-report.
    '''
    def __init__(self, client: KpAnalysisClient, job_id: str):
        '''
        Create a KpAnalysisTaskFuture over a job_id for results retrieval.
        :param client: a client for communicating with the service.
        :param job_id: the job_id. The job_id can be retrieved from the console (it is printed to console when a job is started) or from the user-report.
        '''
        self.client = client
        self.job_id = job_id
        self.polling_timout_secs = 60

    def get_job_id(self) -> str:
        '''
        :return: the job_id
        '''
        return self.job_id

    def get_result(self, top_k_kps: Optional[int] = None, top_k_sentences_per_kp: Optional[int] = None,
                   dont_wait: bool = False, wait_secs: Optional[int] = None, polling_timout_secs: Optional[int] = None,
                   high_verbosity: bool = True):
        '''
        Retreives the job's result. This method polls and waits till the job is done and the result is available.
        The result-json consists:
            * 'keypoint_matchings': a list of keypoint_matching (key point and its matched sentences). Sorted descendingly according to number of matched sentences. each keypoint_matching have:
                * 'keypoint': the key point string.
                * 'matching': a list of matched sentences. each match have the sentences details ('domain', 'comment_id', 'sentence_id', 'sents_in_comment', 'span_start', 'span_end', 'num_tokens', 'argument_quality', 'sentence_text') and a match score ('score') this is the match score between the sentence and the key point. The matchings are sorted descendingly according to their match score.
        :param top_k_kps: use this parameter to truncate the result json to have only the top K key points.
        :param top_k_sentences_per_kp: use this parameter to truncate the result json to have only the top K matched sentences per key point.
        :param dont_wait: when True, tries to get the result once and returns it if it's available, otherwise returns None.
        :param wait_secs: limit the waiting time (in seconds).
        :param polling_timout_secs: sets the time to wait before polling again (in seconds). The default is 60 seconds.
        :param high_verbosity: set to False to reduce the number of messages printed to the logger.
        :return: the key point analysis job result or throws an exception if an error occurs.
        '''
        start_time = time.time()

        do_again = True
        while do_again:
            result = self.client.get_kp_extraction_job_status(self.job_id, top_k_kps=top_k_kps, top_k_sentences_per_kp=top_k_sentences_per_kp)
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
                return result['result']
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
            time.sleep(polling_timout_secs if polling_timout_secs is not None else self.polling_timout_secs)
        return None

    def cancel(self):
        '''
        Cancels (stops) the running job. Please stop unneeded jobs since they use a lot of resources.
        '''
        self.client.cancel_kp_extraction_job(self.job_id)

    def _print_progress_bar(self, progress):
        if 'total_stages' in progress:
            total_stages = progress['total_stages']
            for i in reversed(range(total_stages)):
                stage = str(i + 1)
                stage_i = 'stage_' + stage
                if stage_i in progress and 'inferred_batches' in progress[stage_i] and 'total_batches' in progress[stage_i]:
                    print_progress_bar(progress[stage_i]['inferred_batches'], progress[stage_i]['total_batches'], prefix='Stage %s/%s:' % (stage, str(total_stages)), suffix='Complete', length=50)
                    break
