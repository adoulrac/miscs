“””
backend_k8s.py — FastAPI K8s Terminal
Sessions shell persistantes : le /bin/sh reste vivant entre les commandes.
Chaque session a un ID, un CWD suivi, et un shell ouvert dans le pod.
“””

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from kubernetes import client, config
from kubernetes.stream import stream as k8s_stream
import uuid
import threading
import queue
import time

app = FastAPI(title=“K8s Terminal API”)

# ── Kubeconfig ─────────────────────────────────────────────────────────────────

try:
config.load_kube_config()
except Exception:
config.load_incluster_config()

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
return {
“pods”: [
{
“name”: p.metadata.name,
“status”: p.status.phase,
“containers”: [c.name for c in p.spec.containers],
“ready”: all(cs.ready for cs in (p.status.container_statuses or [])),
}
for p in pods.items
]
}

# ══════════════════════════════════════════════════════════════════════════════

# SESSIONS — shell persistant par session_id

# ══════════════════════════════════════════════════════════════════════════════

class ShellSession:
“””
Maintient un process /bin/sh ouvert dans un pod K8s.
Les commandes sont envoyées via stdin, la sortie est lue en continu
dans un thread dédié et mise en queue pour le streaming SSE.
“””

```
_END_MARKER = "__CMD_DONE__"

def __init__(self, namespace: str, pod: str, container: str):
    self.session_id = str(uuid.uuid4())
    self.namespace  = namespace
    self.pod        = pod
    self.container  = container
    self.cwd        = "~"
    self.alive      = False
    self._resp      = None
    self._out_queue: queue.Queue = queue.Queue()
    self._reader_thread: threading.Thread | None = None

def open(self):
    """Ouvre le shell dans le pod et démarre le thread de lecture."""
    v1 = client.CoreV1Api()
    self._resp = k8s_stream(
        v1.connect_get_namespaced_pod_exec,
        self.pod,
        self.namespace,
        container=self.container,
        command=["/bin/sh"],
        stderr=True,
        stdin=True,
        stdout=True,
        tty=False,
        _preload_content=False,
    )
    self.alive = True

    self._reader_thread = threading.Thread(target=self._reader, daemon=True)
    self._reader_thread.start()

    # Supprime le prompt PS1 et initialise le CWD
    self._write("export PS1=''\n")
    time.sleep(0.15)
    self._drain_queue()

    # Récupère le CWD initial
    self._write("pwd\necho __PWD_DONE__\n")
    cwd_lines = []
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            k, v = self._out_queue.get(timeout=1)
            if k == "stdout" and v.strip() == "__PWD_DONE__":
                if cwd_lines:
                    self.cwd = cwd_lines[-1].strip()
                break
            if k == "stdout" and v.strip():
                cwd_lines.append(v)
        except queue.Empty:
            break

def _reader(self):
    """Thread de lecture continue stdout/stderr → queue."""
    try:
        while self._resp.is_open():
            self._resp.update(timeout=1)
            if self._resp.peek_stdout():
                for line in self._resp.read_stdout().splitlines():
                    self._out_queue.put(("stdout", line))
            if self._resp.peek_stderr():
                for line in self._resp.read_stderr().splitlines():
                    self._out_queue.put(("stderr", line))
    except Exception as e:
        self._out_queue.put(("stderr", f"[session error] {e}"))
    finally:
        self.alive = False
        self._out_queue.put(("__CLOSED__", ""))

def _write(self, text: str):
    if self._resp and self._resp.is_open():
        self._resp.write_stdin(text)

def _drain_queue(self, timeout: float = 0.3):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            self._out_queue.get_nowait()
        except queue.Empty:
            break

def run(self, command: str):
    """
    Exécute une commande et génère la sortie ligne par ligne.
    Met à jour self.cwd après chaque commande.
    Yields: (type, text) avec type = 'stdout' | 'stderr' | '__END__'
    """
    if not self.alive:
        yield ("stderr", "[session terminée — reconnectez-vous]")
        yield ("__END__", "")
        return

    # Envoie commande + marqueur de fin
    self._write(f"{command}\necho {self._END_MARKER}\n")

    while True:
        try:
            kind, text = self._out_queue.get(timeout=30)
        except queue.Empty:
            yield ("stderr", "[timeout — pas de réponse du pod]")
            break

        if kind == "__CLOSED__":
            self.alive = False
            break

        # Marqueur de fin détecté → on récupère le CWD
        if kind == "stdout" and text.strip() == self._END_MARKER:
            self._write("pwd\necho __PWD_DONE__\n")
            cwd_buf = []
            while True:
                try:
                    k2, v2 = self._out_queue.get(timeout=5)
                except queue.Empty:
                    break
                if k2 == "stdout" and v2.strip() == "__PWD_DONE__":
                    if cwd_buf:
                        self.cwd = cwd_buf[-1].strip()
                    break
                if k2 == "stdout" and v2.strip():
                    cwd_buf.append(v2)
            break

        if text.strip() == "":
            continue

        yield (kind, text)

    yield ("__END__", "")

def close(self):
    if self._resp and self._resp.is_open():
        try:
            self._write("exit\n")
            time.sleep(0.1)
            self._resp.close()
        except Exception:
            pass
    self.alive = False
```

# ── Registre des sessions ──────────────────────────────────────────────────────

_sessions: dict[str, ShellSession] = {}
_sessions_lock = threading.Lock()

def get_session(session_id: str) -> ShellSession:
with _sessions_lock:
sess = _sessions.get(session_id)
if not sess or not sess.alive:
raise HTTPException(status_code=404, detail=f”Session {session_id} introuvable ou expirée”)
return sess

# ══════════════════════════════════════════════════════════════════════════════

# ENDPOINTS SESSION

# ══════════════════════════════════════════════════════════════════════════════

class OpenSessionRequest(BaseModel):
namespace: str
pod: str
container: str

@app.post(”/session/open”)
def open_session(req: OpenSessionRequest):
“”“Ouvre un shell persistant dans un pod. Retourne un session_id.”””
sess = ShellSession(req.namespace, req.pod, req.container)
try:
sess.open()
except Exception as e:
raise HTTPException(status_code=500, detail=str(e))
with _sessions_lock:
_sessions[sess.session_id] = sess
return {
“session_id”: sess.session_id,
“cwd”: sess.cwd,
“pod”: req.pod,
“container”: req.container,
“namespace”: req.namespace,
}

@app.delete(”/session/{session_id}”)
def close_session(session_id: str):
“”“Ferme et supprime une session.”””
with _sessions_lock:
sess = _sessions.pop(session_id, None)
if sess:
sess.close()
return {“closed”: session_id}

@app.get(”/session/{session_id}/status”)
def session_status(session_id: str):
with _sessions_lock:
sess = _sessions.get(session_id)
if not sess:
return {“alive”: False, “cwd”: None}
return {“alive”: sess.alive, “cwd”: sess.cwd}

# ══════════════════════════════════════════════════════════════════════════════

# EXEC — session persistante, streaming SSE

# ══════════════════════════════════════════════════════════════════════════════

class RunRequest(BaseModel):
session_id: str
command: str

@app.post(”/exec”)
def exec_in_session(req: RunRequest):
“””
Exécute une commande dans la session persistante.
cd fonctionne : le CWD est maintenu entre les appels.

```
Format SSE :
  data: stdout:<ligne>
  data: stderr:<ligne>
  data: cwd:<chemin>     ← CWD mis à jour, envoyé avant __END__
  data: __END__
"""
sess = get_session(req.session_id)

def generator():
    for kind, text in sess.run(req.command):
        if kind == "__END__":
            yield f"data: cwd:{sess.cwd}\n\n"
            yield "data: __END__\n\n"
            return
        yield f"data: {kind}:{text}\n\n"

return StreamingResponse(
    generator(),
    media_type="text/event-stream",
    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
)
```

# ══════════════════════════════════════════════════════════════════════════════

# LOGS — streaming SSE

# ══════════════════════════════════════════════════════════════════════════════

@app.get(”/logs/{namespace}/{pod}”)
def stream_logs(
namespace: str,
pod: str,
container: str | None = None,
tail_lines: int = 100,
follow: bool = False,
):
v1 = client.CoreV1Api()

```
def generator():
    try:
        kwargs = dict(
            name=pod, namespace=namespace,
            tail_lines=tail_lines, follow=follow,
            _preload_content=False, timestamps=True,
        )
        if container:
            kwargs["container"] = container
        for line in v1.read_namespaced_pod_log(**kwargs):
            text = line.decode("utf-8") if isinstance(line, bytes) else line
            yield f"data: {text.rstrip()}\n\n"
    except client.exceptions.ApiException as e:
        yield f"data: [ERROR] {e}\n\n"
    except GeneratorExit:
        pass

return StreamingResponse(
    generator(),
    media_type="text/event-stream",
    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
)
```