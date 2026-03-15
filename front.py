“””
backend_k8s.py — FastAPI endpoints pour le terminal K8s
Gère : listing namespaces/pods, exec one-shot, streaming logs
“””

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from kubernetes import client, config
from kubernetes.stream import stream
import asyncio
import time

app = FastAPI(title=“K8s Terminal API”)

# ── Chargement kubeconfig ──────────────────────────────────────────────────────

try:
config.load_kube_config()          # local (~/.kube/config)
except Exception:
config.load_incluster_config()     # dans un pod K8s

# ══════════════════════════════════════════════════════════════════════════════

# LISTING

# ══════════════════════════════════════════════════════════════════════════════

@app.get(”/namespaces”)
def list_namespaces():
“”“Retourne la liste des namespaces.”””
v1 = client.CoreV1Api()
ns_list = v1.list_namespace()
return {“namespaces”: [ns.metadata.name for ns in ns_list.items]}

@app.get(”/namespaces/{namespace}/pods”)
def list_pods(namespace: str):
“”“Retourne les pods d’un namespace avec leur statut.”””
v1 = client.CoreV1Api()
try:
pods = v1.list_namespaced_pod(namespace)
except client.exceptions.ApiException as e:
raise HTTPException(status_code=e.status, detail=str(e))

```
result = []
for p in pods.items:
    containers = [c.name for c in p.spec.containers]
    result.append({
        "name": p.metadata.name,
        "status": p.status.phase,
        "containers": containers,
        "ready": all(
            cs.ready
            for cs in (p.status.container_statuses or [])
        ),
    })
return {"pods": result}
```

# ══════════════════════════════════════════════════════════════════════════════

# EXEC ONE-SHOT

# ══════════════════════════════════════════════════════════════════════════════

class ExecRequest(BaseModel):
namespace: str
pod: str
container: str
command: str          # commande shell libre, ex: “ls -la /app”

@app.post(”/exec”)
def exec_command(req: ExecRequest):
“””
Exécute une commande dans un pod et retourne stdout + stderr.
Usage : commandes courtes dont on veut la réponse complète.
“””
v1 = client.CoreV1Api()
try:
resp = stream(
v1.connect_get_namespaced_pod_exec,
req.pod,
req.namespace,
container=req.container,
command=[”/bin/sh”, “-c”, req.command],
stderr=True,
stdin=False,
stdout=True,
tty=False,
_preload_content=True,       # attend la fin avant de retourner
)
# kubernetes-client mélange stdout/stderr dans resp quand preload=True
# On sépare proprement avec _preload_content=False si besoin,
# mais pour le one-shot c’est suffisant.
return {
“stdout”: resp,
“stderr”: “”,
“exit_code”: 0,
}
except client.exceptions.ApiException as e:
raise HTTPException(status_code=e.status, detail=str(e))
except Exception as e:
return {“stdout”: “”, “stderr”: str(e), “exit_code”: 1}

# ══════════════════════════════════════════════════════════════════════════════

# LOGS STREAMING  (Server-Sent Events)

# ══════════════════════════════════════════════════════════════════════════════

@app.get(”/logs/{namespace}/{pod}”)
def stream_logs(
namespace: str,
pod: str,
container: str | None = None,
tail_lines: int = 100,
follow: bool = False,
):
“””
Stream les logs d’un pod via Server-Sent Events.
- tail_lines : nb de lignes historiques à envoyer
- follow=true : continue à streamer les nouvelles lignes (comme kubectl logs -f)

```
Côté Streamlit : requests.get(..., stream=True) + iter_lines()
"""
v1 = client.CoreV1Api()

def log_generator():
    try:
        kwargs = dict(
            name=pod,
            namespace=namespace,
            tail_lines=tail_lines,
            follow=follow,
            _preload_content=False,   # IMPORTANT : stream ligne par ligne
            timestamps=True,
        )
        if container:
            kwargs["container"] = container

        w = v1.read_namespaced_pod_log(**kwargs)

        for line in w:
            # SSE format : "data: <contenu>\n\n"
            text = line.decode("utf-8") if isinstance(line, bytes) else line
            yield f"data: {text.rstrip()}\n\n"

    except client.exceptions.ApiException as e:
        yield f"data: [ERROR] {e}\n\n"
    except GeneratorExit:
        pass   # client déconnecté, on arrête proprement

return StreamingResponse(
    log_generator(),
    media_type="text/event-stream",
    headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",   # désactive le buffer nginx si proxy
    },
)
```

# ══════════════════════════════════════════════════════════════════════════════

# EXEC STREAMING  (pour les commandes longues)

# ══════════════════════════════════════════════════════════════════════════════

@app.post(”/exec/stream”)
def exec_stream(req: ExecRequest):
“””
Exécute une commande et streame la sortie ligne par ligne (SSE).
Usage : commandes longues (build, migration, scripts…)
“””
v1 = client.CoreV1Api()

```
def exec_generator():
    try:
        resp = stream(
            v1.connect_get_namespaced_pod_exec,
            req.pod,
            req.namespace,
            container=req.container,
            command=["/bin/sh", "-c", req.command],
            stderr=True,
            stdin=False,
            stdout=True,
            tty=False,
            _preload_content=False,   # stream ligne par ligne
        )
        while resp.is_open():
            resp.update(timeout=1)
            if resp.peek_stdout():
                out = resp.read_stdout()
                for line in out.splitlines():
                    yield f"data: stdout:{line}\n\n"
            if resp.peek_stderr():
                err = resp.read_stderr()
                for line in err.splitlines():
                    yield f"data: stderr:{line}\n\n"

        yield "data: __END__\n\n"

    except Exception as e:
        yield f"data: stderr:{str(e)}\n\n"
        yield "data: __END__\n\n"

return StreamingResponse(
    exec_generator(),
    media_type="text/event-stream",
    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
)
```