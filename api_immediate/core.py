"""Independent jobs: one thread per submission and per HTTP request, no worker pool."""
import copy
import threading
import time
import uuid


class JobManager:
    def __init__(self, prepare, save):
        self.prepare = prepare
        self.save = save
        self.jobs = {}
        self.lock = threading.RLock()

    def submit(self, owner, payload):
        snapshot = copy.deepcopy(payload)
        job_id = uuid.uuid4().hex
        job = dict(id=job_id, owner=owner, node_id=snapshot['target'],
                   client_ref=snapshot.get('client_ref', ''),
                   node_type=snapshot['prompt'][snapshot['target']]['class_type'],
                   created=time.time(), status='preparing', items=[], error='')
        with self.lock:
            # Retain active jobs and the latest 200 finished jobs only.
            finished = [k for k, v in self.jobs.items()
                        if v['status'] in ('completed', 'partial', 'failed', 'cancelled')]
            for key in finished[:-199]:
                self.jobs.pop(key, None)
            self.jobs[job_id] = job
        try:
            threading.Thread(target=self._run, args=(job, snapshot), daemon=True,
                             name=f'api-immediate-{job_id[:8]}').start()
        except Exception:
            with self.lock:
                self.jobs.pop(job_id, None)
            raise
        return job_id

    def list(self, owner):
        with self.lock:
            return copy.deepcopy([{k: v for k, v in job.items() if k != 'owner'}
                                  for job in self.jobs.values() if job['owner'] == owner])

    def cancel(self, owner, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if not job or job['owner'] != owner:
                return False
            if job['status'] in ('preparing', 'running'):
                job['status'] = 'cancelled'
                for item in job['items']:
                    if item['status'] == 'running':
                        item['status'] = 'cancelled'
            return True

    def _run(self, job, snapshot):
        try:
            count, request_one, secrets = self.prepare(snapshot)
            with self.lock:
                if job['status'] == 'cancelled':
                    return
                job['items'] = [dict(index=i, status='running', images=[], error='')
                                for i in range(count)]
                job['status'] = 'running'
            workers = []
            for index in range(count):
                with self.lock:
                    if job['status'] == 'cancelled':
                        break
                worker = threading.Thread(target=self._one,
                    args=(job, index, request_one, secrets), daemon=True)
                try:
                    worker.start()
                    workers.append(worker)
                except Exception as exc:
                    with self.lock:
                        job['items'][index].update(status='failed', error=type(exc).__name__)
            for worker in workers:
                worker.join()
            with self.lock:
                if job['status'] != 'cancelled':
                    good = sum(i['status'] == 'completed' for i in job['items'])
                    job['status'] = 'completed' if good == count else 'partial' if good else 'failed'
        except Exception as exc:
            with self.lock:
                if job['status'] != 'cancelled':
                    # Do not echo arbitrary input values or credentials from preparation errors.
                    job.update(status='failed', error=f'准备失败：{type(exc).__name__}。请检查节点参数、参考图和 API Key。')

    def _one(self, job, index, request_one, secrets):
        try:
            with self.lock:
                if job['status'] == 'cancelled':
                    return
            image, error = request_one(index)
            if error or image is None:
                raise RuntimeError(error or '接口未返回图片')
            with self.lock:
                if job['status'] == 'cancelled':
                    return
            # Unique filenames per job/request; image encoding never holds the manager lock.
            images = self.save(image, job['id'], index)
            with self.lock:
                if job['status'] != 'cancelled':
                    job['items'][index].update(status='completed', images=images)
        except Exception as exc:
            message = str(exc)
            lower = message.lower()
            if 'insufficient_user_quota' in lower or 'insufficient_quota' in lower or ('quota' in lower and 'not enough' in lower):
                message = 'API 额度不足：请检查服务商账户余额及 API Key 配额。\n' + message
            for secret in secrets:
                if secret:
                    message = message.replace(secret, '[redacted]')
            with self.lock:
                if job['status'] != 'cancelled':
                    job['items'][index].update(status='failed', error=message[:1500])
