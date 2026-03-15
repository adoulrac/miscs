“””
backend_k8s.py — FastAPI endpoints pour le terminal K8s
Tout passe par le streaming (SSE), pas de one-shot.
“””

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from kubernetes import client, config
from kubernetes.stream import stream

app = FastAPI(title=“K8s Terminal API”)

# ── Chargement kubeconfig ──────────────────────────────────────────────────────

try:
config.load_kube_config()       # local (~/.kube/config)
except Exception:
config.load_incluster_config()  # dans un pod K8s

# ══════════════════════════════════════════════════════════════════════════════

# LISTING

# ══════════════════════════════════════════════════════════════════════════════

@app.get(”/namespaces”)
def list_namespaces():
v1 = client.CoreV1Api()
return {“namespaces”: [ns.metadata.name for ns in v1.list_namespace().items]}

@app.get(”/namespaces/{namespace}/pods”)
def list_pods(namespace: str):
v1 = client.CoreV1Api()
try:
pods = v1.list_namespaced_pod(namespace)
except client.exceptions.ApiException as e:
raise HTTPException(status_code=e.status, detail=str(e))

```
return {
    "pods": [
        {
            "name": p.metadata.name,
            "status": p.status.phase,
            "containers": [c.name for c in p.spec.containers],
            "ready": all(cs.ready for cs in (p.status.container_statuses or [])),
        }
        for p in pods.items
    ]
}
```

# ══════════════════════════════════════════════════════════════════════════════

# EXEC — toujours en streaming (SSE)

# ══════════════════════════════════════════════════════════════════════════════

class ExecRequest(BaseModel):
namespace: str
pod: str
container: str
command: str   # commande shell libre, ex: “ls -la /app”

@app.post(”/exec”)
def exec_stream(req: ExecRequest):
“””
Exécute une commande dans un pod et streame la sortie via SSE.
Fonctionne aussi bien pour les commandes courtes que longues.

```
Format SSE :
  data: stdout:<ligne>
  data: stderr:<ligne>
  data: __END__          ← signal de fin
"""
v1 = client.CoreV1Api()

def generator():
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
            _preload_content=False,  # indispensable pour le streaming
        )
        while resp.is_open():
            resp.update(timeout=1)
            if resp.peek_stdout():
                for line in resp.read_stdout().splitlines():
                    yield f"data: stdout:{line}\n\n"
            if resp.peek_stderr():
                for line in resp.read_stderr().splitlines():
                    yield f"data: stderr:{line}\n\n"

        yield "data: __END__\n\n"

    except client.exceptions.ApiException as e:
        yield f"data: stderr:{e}\n\n"
        yield "data: __END__\n\n"
    except Exception as e:
        yield f"data: stderr:{e}\n\n"
        yield "data: __END__\n\n"

return StreamingResponse(
    generator(),
    media_type="text/event-stream",
    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
)
```

# ══════════════════════════════════════════════════════════════════════════════

# LOGS — streaming (SSE)

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
Stream les logs d’un pod.
- tail_lines : nb de lignes historiques
- follow=true : continue à streamer (équivalent kubectl logs -f)
“””
v1 = client.CoreV1Api()

```
def generator():
    try:
        kwargs = dict(
            name=pod,
            namespace=namespace,
            tail_lines=tail_lines,
            follow=follow,
            _preload_content=False,
            timestamps=True,
        )
        if container:
            kwargs["container"] = container

        for line in v1.read_namespaced_pod_log(**kwargs):
            text = line.decode("utf-8") if isinstance(line, bytes) else line
            yield f"data: {text.rstrip()}\n\n"

    except client.exceptions.ApiException as e:
        yield f"data: [ERROR] {e}\n\n"
    except GeneratorExit:
        pass  # client déconnecté

return StreamingResponse(
    generator(),
    media_type="text/event-stream",
    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
)
```