import datetime
import logging

from typing import Optional, List
from debater_python_api.api.clients.keypoints_client import KpAnalysisClient

admin_reports_endpoint = '/admin_report'
admin_actions_endpoint = '/admin_action'

class KpAnalysisAdminClient(KpAnalysisClient):
    def __init__(self, admin_password, apikey: str, host: Optional[str]=None, verify_certificate: bool = True):
        super().__init__(apikey, host, verify_certificate)
        self.admin_password = admin_password

    def get_admin_password_header(self, admin_password):
        return {'admin-password': self.admin_password}

    def admin_report_get_domain_statuses(self):
        res = self._get(self.host + admin_reports_endpoint, {'report': 'domain_statuses'},
                        headers=self.get_admin_password_header(self.admin_password))
        logging.info(f'n_domain_statuses: {len(res["domain_statuses"])}')
        return res

    def admin_report_get_job_statuses_by_days(self, since_days_ago: int, till_days_ago: Optional[int] = None):
        res = self._get(self.host + admin_reports_endpoint,
                        {'report': 'job_statuses_by_days', 'since_days_ago': since_days_ago,
                         'till_days_ago': till_days_ago},
                        headers=self.get_admin_password_header(self.admin_password))
        logging.info(f'n_job_statuses between {since_days_ago} and {till_days_ago}: {len(res["jobs_statuses"])}')
        return res

    def admin_report_get_job_statuses_by_dates(self, since_date: datetime, till_date=None):
        since_date_str = since_date.strftime("%m/%d/%Y, %H:%M:%S")
        till_date_str = None if till_date is None else till_date.strftime("%m/%d/%Y, %H:%M:%S")
        res = self._get(self.host + admin_reports_endpoint,
                        {'report': 'job_statuses_by_dates', 'since_date': since_date_str, 'till_date': till_date_str},
                        headers=self.get_admin_password_header(self.admin_password))
        logging.info(f'n_job_statuses between {since_date_str} and {till_date_str}: {len(res["jobs_statuses"])}')
        return res

    def admin_report_get_not_finished_job_statuses(self):
        res = self._get(self.host + admin_reports_endpoint, {'report': 'not_finished_job_statuses'},
                        headers=self.get_admin_password_header(self.admin_password))
        logging.info(f'not_finished_job_statuses: {len(res["jobs_statuses"])}')
        return res

    def update_jobs_statuses_by_ids(self, job_ids: List[str]):
        print(job_ids)
        res = self._get(self.host + admin_reports_endpoint,
                        {'report': 'update_jobs_statuses_by_ids', 'job_ids': job_ids},
                        headers=self.get_admin_password_header(self.admin_password))
        logging.info(f'n_job_statuses for {len(job_ids)} job_ids: {len(res["jobs_statuses"])}')
        return res

    def admin_report_get_not_finished_comment_batches(self):
        res = self._get(self.host + admin_reports_endpoint,
                        {'report': 'comment_batches_statuses'},
                        headers=self.get_admin_password_header(self.admin_password))
        logging.info(f'not_done_comment_batches: {len(res)}')
        return res

    def admin_report_get_comments_stats(self, user_id=None, since_date=None, till_date=None):
        params = {'admin_password': self.admin_password, 'report': 'comments_stats'}
        if user_id is not None:
            params['user_id'] = str(user_id)
        if since_date is not None:
            params['since_date'] = since_date.strftime("%m/%d/%Y, %H:%M:%S")
        if till_date is not None:
            params['till_date'] = till_date.strftime("%m/%d/%Y, %H:%M:%S")

        res = self._get(self.host + admin_reports_endpoint, params,
                        headers=self.get_admin_password_header(self.admin_password))
        logging.info(f'comments_stats: {len(res)}')
        return res

    def admin_action_delete_user(self, user_id: str):
        res = self._get(self.host + admin_actions_endpoint,
                        {'action': 'delete_user', 'user_id': user_id},
                        headers=self.get_admin_password_header(self.admin_password))
        logging.info(f'delete_users: {res}')
        return res

    def admin_action_delete_user_domain(self, user_id: str, domain: str):
        res = self._get(self.host + admin_actions_endpoint,
                        {'action': 'delete_user_domain', 'user_id': user_id, 'domain': domain},
                        headers=self.get_admin_password_header(self.admin_password))
        logging.info(f'delete_user_domain: {res}')
        return res

    def admin_action_delete_old_domains_by_date(self, older_than_date: datetime):
        older_than_date_str = older_than_date.strftime("%m/%d/%Y, %H:%M:%S")
        res = self._get(self.host + admin_actions_endpoint,
                        {'action': 'delete_old_domains',
                         'older_than_date': older_than_date_str},
                        headers=self.get_admin_password_header(self.admin_password))
        logging.info(f'deleted jobs older than {older_than_date_str}')
        return res

    def admin_action_cancel_job(self, job_id: str):
        res = self._get(self.host + admin_actions_endpoint,
                        {'action': 'cancel_job', 'job_id': job_id},
                        headers=self.get_admin_password_header(self.admin_password))
        logging.info(f'cancel_job: {res}')
        return res

    def admin_action_cancel_all_jobs(self):
        res = self._get(self.host + admin_actions_endpoint,
                        {'action': 'cancel_all_jobs'},
                        headers=self.get_admin_password_header(self.admin_password))
        logging.info(f'cancel_all_jobs: {res}')
        return res

    def admin_action_cancel_all_jobs_by_user(self, user_id: str):
        res = self._get(self.host + admin_actions_endpoint,
                        {'action': 'cancel_all_jobs_by_user', 'user_id': user_id},
                        headers=self.get_admin_password_header(self.admin_password))
        logging.info(f'cancel_all_jobs_by_user: {res}')
        return res

    def admin_action_cancel_all_jobs_by_domain(self, user_id: str, domain: str):
        res = self._get(self.host + admin_actions_endpoint,
                        {'action': 'cancel_all_jobs_by_domain', 'user_id': user_id, 'domain': domain},
                        headers=self.get_admin_password_header(self.admin_password))
        logging.info(f'cancel_all_jobs_by_domain: {res}')
        return res

    def admin_action_set_user_limit(self, user_id: str, user_limit: int):
        res = self._get(self.host + admin_actions_endpoint,
                        {'action': 'set_user_limit', 'user_id': user_id, 'user_limit': user_limit},
                        headers=self.get_admin_password_header(self.admin_password))
        logging.info(f'set_user_limit: {res}')
        return res
