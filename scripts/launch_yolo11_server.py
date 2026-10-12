"""Reserve a TCP endpoint before loading the LLM, then serve on that socket."""
import errno
import json
import os
from pathlib import Path
import socket
import sys


def reserve_socket(host, preferred_port):
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind((host, preferred_port))
    except OSError as error:
        if error.errno != errno.EADDRINUSE:
            listener.close()
            raise
        try:
            listener.bind((host, 0))
        except OSError:
            listener.close()
            raise
        print(f'Port {preferred_port} is occupied; reserved port {listener.getsockname()[1]} instead.', flush=True)
    return listener


def launch(repo=None, app='server:app'):
    repo = Path(repo) if repo else Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    host = os.environ.get('SERVER_HOSTNAME', socket.gethostname())
    preferred = int(os.environ.get('LLMGE_PORT', '8293'))
    job = os.environ['SLURM_JOB_ID']
    with reserve_socket(host, preferred) as listener:
        port = listener.getsockname()[1]
        os.environ['LLMGE_PORT'] = str(port)
        connection = {'hostname': host, 'port': port, 'slurm_job_id': job}
        for marker in (repo / f'run_job_outputs/server/connection-{job}.json',
                       repo / 'llm-server-connection.json'):
            marker.parent.mkdir(parents=True, exist_ok=True)
            temporary = marker.with_suffix(f'.{os.getpid()}.tmp')
            temporary.write_text(json.dumps(connection, indent=2))
            temporary.replace(marker)
        print(f'Reserved LLM endpoint http://{host}:{port}/ for job {job}', flush=True)
        # Passing the actual socket keeps it reserved throughout model loading.
        import uvicorn
        server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, workers=1))
        server.run(sockets=[listener])


if __name__ == '__main__':
    launch()
