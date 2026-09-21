from aiohttp import web
import re
import nodes
import folder_paths
from server import PromptServer
from .core import JobManager
from .backend import validate, prepare, save_images


manager = JobManager(lambda data: prepare(data, nodes.NODE_CLASS_MAPPINGS),
    lambda image, job, index: save_images(image, job, index, folder_paths.get_output_directory()))


def owner(request):
    value = request.headers.get('X-API-Immediate-Session', '')
    if not re.fullmatch(r'[a-f0-9]{32}', value):
        raise web.HTTPBadRequest(text='Missing session token')
    return value


@PromptServer.instance.routes.post('/api-immediate/jobs')
async def submit(request):
    session = owner(request)
    try:
        data = await request.json()
        if not isinstance(data, dict):
            raise ValueError('请求格式无效')
        payload = validate(data, nodes.NODE_CLASS_MAPPINGS)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        return web.json_response({'error': str(exc)}, status=400)
    job_id = manager.submit(session, payload)
    return web.json_response({'id': job_id}, status=202)


@PromptServer.instance.routes.get('/api-immediate/jobs')
async def list_jobs(request):
    return web.json_response({'jobs': manager.list(owner(request))}, headers={'Cache-Control': 'no-store'})


@PromptServer.instance.routes.post('/api-immediate/jobs/{job_id}/cancel')
async def cancel(request):
    found = manager.cancel(owner(request), request.match_info['job_id'])
    return web.json_response({'ok': found}, status=200 if found else 404)
