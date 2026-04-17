import time
from datetime import datetime
from typing import Optional

import clickhouse_connect
from croniter import croniter
from loguru import logger


# -----------------------------
# CONFIG
# -----------------------------

CLICKHOUSE_HOST = "localhost"
CLICKHOUSE_PORT = 8123
CLICKHOUSE_USER = "default"
CLICKHOUSE_PASSWORD = ""
CLICKHOUSE_DATABASE = "default"

POLL_INTERVAL_SECONDS = 60


# -----------------------------
# LOGURU SETUP
# -----------------------------

logger.remove()

logger.add(
    "mart_runner.log",
    rotation="100 MB",
    retention="10 days",
    compression="zip",
    enqueue=True,
    backtrace=True,
    diagnose=True,
)

logger.add(
    lambda msg: print(msg, end=""),
    level="INFO",
)


# -----------------------------
# CLICKHOUSE
# -----------------------------

def get_client():
    return clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        username=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASSWORD,
        database=CLICKHOUSE_DATABASE,
    )


# -----------------------------
# CONFIG TABLE
# -----------------------------

def ensure_config_table(client):

    client.command("""
    CREATE TABLE IF NOT EXISTS mart_config
    (
        mart_name String,
        target_table String,
        query String,

        schedule_cron String,

        chunk_column Nullable(String),
        chunk_size Nullable(UInt64),

        enabled UInt8 DEFAULT 1,

        last_run DateTime DEFAULT toDateTime(0),

        created_at DateTime DEFAULT now()
    )
    ENGINE = MergeTree
    ORDER BY mart_name
    """)


# -----------------------------
# EXECUTION
# -----------------------------

def run_simple_insert(client, target_table, query):

    sql = f"""
    INSERT INTO {target_table}
    {query}
    """

    logger.info(f"Running mart insert into {target_table}")

    client.command(sql)


def run_chunked_insert(client, target_table, base_query, chunk_column, chunk_size):

    logger.info(
        f"Running chunked mart column={chunk_column} size={chunk_size}"
    )

    minmax_query = f"""
    SELECT
        min({chunk_column}),
        max({chunk_column})
    FROM ({base_query})
    """

    min_val, max_val = client.query(minmax_query).result_rows[0]

    if min_val is None:
        logger.info("No data to process")
        return

    start = min_val

    while start <= max_val:

        end = start + chunk_size

        sql = f"""
        INSERT INTO {target_table}
        SELECT *
        FROM ({base_query})
        WHERE {chunk_column} >= {start}
        AND {chunk_column} < {end}
        """

        logger.info(f"Processing chunk {start} -> {end}")

        client.command(sql)

        start = end


def run_mart(client, row):

    (
        mart_name,
        target_table,
        query,
        cron,
        chunk_column,
        chunk_size,
        last_run
    ) = row

    logger.info(f"Starting mart {mart_name}")

    if chunk_column and chunk_size:

        run_chunked_insert(
            client,
            target_table,
            query,
            chunk_column,
            chunk_size
        )

    else:

        run_simple_insert(
            client,
            target_table,
            query
        )

    client.command(f"""
        ALTER TABLE mart_config
        UPDATE last_run = now()
        WHERE mart_name = '{mart_name}'
    """)

    logger.success(f"Finished mart {mart_name}")


# -----------------------------
# SCHEDULER
# -----------------------------

def fetch_configs(client):

    rows = client.query("""
        SELECT
            mart_name,
            target_table,
            query,
            schedule_cron,
            chunk_column,
            chunk_size,
            last_run
        FROM mart_config
        WHERE enabled = 1
    """).result_rows

    return rows


def should_run(cron_expr, last_run):

    now = datetime.utcnow()

    itr = croniter(cron_expr, last_run)

    next_run = itr.get_next(datetime)

    return next_run <= now


# -----------------------------
# MAIN LOOP
# -----------------------------

def scheduler():

    client = get_client()

    ensure_config_table(client)

    logger.info("Mart runner started")

    while True:

        try:

            rows = fetch_configs(client)

            for row in rows:

                (
                    mart_name,
                    target_table,
                    query,
                    cron,
                    chunk_column,
                    chunk_size,
                    last_run
                ) = row

                if should_run(cron, last_run):

                    try:

                        run_mart(client, row)

                    except Exception:

                        logger.exception(
                            f"Mart {mart_name} failed"
                        )

        except Exception:

            logger.exception("Scheduler loop error")

        time.sleep(POLL_INTERVAL_SECONDS)


# -----------------------------
# ENTRYPOINT
# -----------------------------

if __name__ == "__main__":

    scheduler()










from kubernetes import client, config
from collections import defaultdict


def parse_cpu(cpu):
    if cpu.endswith("n"):
        return int(cpu[:-1]) / 1e9
    if cpu.endswith("u"):
        return int(cpu[:-1]) / 1e6
    if cpu.endswith("m"):
        return int(cpu[:-1]) / 1000
    return float(cpu)


def parse_memory(mem):
    units = {
        "Ki": 1024,
        "Mi": 1024**2,
        "Gi": 1024**3,
        "Ti": 1024**4
    }

    for unit in units:
        if mem.endswith(unit):
            return int(mem[:-len(unit)]) * units[unit]

    return int(mem)


def get_pod_metrics():
    api = client.CustomObjectsApi()

    metrics = api.list_cluster_custom_object(
        group="metrics.k8s.io",
        version="v1beta1",
        plural="pods"
    )

    usage_map = {}

    for pod in metrics["items"]:
        ns = pod["metadata"]["namespace"]
        name = pod["metadata"]["name"]

        cpu = 0
        memory = 0

        for container in pod["containers"]:
            cpu += parse_cpu(container["usage"]["cpu"])
            memory += parse_memory(container["usage"]["memory"])

        usage_map[(ns, name)] = {
            "cpu_usage": cpu,
            "memory_usage": memory
        }

    return usage_map


def get_cluster_node_usage():

    v1 = client.CoreV1Api()

    pods = v1.list_pod_for_all_namespaces().items
    nodes = v1.list_node().items

    usage_map = get_pod_metrics()

    node_data = {}

    for node in nodes:
        node_name = node.metadata.name

        node_data[node_name] = {
            "capacity": {
                "cpu": node.status.capacity.get("cpu"),
                "memory": node.status.capacity.get("memory"),
            },
            "allocatable": {
                "cpu": node.status.allocatable.get("cpu"),
                "memory": node.status.allocatable.get("memory"),
            },
            "pods": []
        }

    for pod in pods:

        if not pod.spec.node_name:
            continue

        node = pod.spec.node_name

        pod_cpu_request = 0
        pod_cpu_limit = 0
        pod_mem_request = 0
        pod_mem_limit = 0

        for container in pod.spec.containers:

            req = container.resources.requests or {}
            lim = container.resources.limits or {}

            if "cpu" in req:
                pod_cpu_request += parse_cpu(req["cpu"])

            if "memory" in req:
                pod_mem_request += parse_memory(req["memory"])

            if "cpu" in lim:
                pod_cpu_limit += parse_cpu(lim["cpu"])

            if "memory" in lim:
                pod_mem_limit += parse_memory(lim["memory"])

        usage = usage_map.get(
            (pod.metadata.namespace, pod.metadata.name),
            {"cpu_usage": 0, "memory_usage": 0}
        )

        pod_info = {
            "namespace": pod.metadata.namespace,
            "name": pod.metadata.name,
            "cpu_request": pod_cpu_request,
            "cpu_limit": pod_cpu_limit,
            "memory_request": pod_mem_request,
            "memory_limit": pod_mem_limit,
            "cpu_usage": usage["cpu_usage"],
            "memory_usage": usage["memory_usage"]
        }

        node_data[node]["pods"].append(pod_info)

    return node_data


if __name__ == "__main__":

    # Load config automatically (works locally and in cluster)
    try:
        config.load_kube_config()
    except:
        config.load_incluster_config()

    data = get_cluster_node_usage()

    import json
    print(json.dumps(data, indent=2))





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