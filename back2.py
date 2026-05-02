# =========================================================
# DATAMART ENGINE — COMPOSITE CHUNKING VERSION (PROD)
# =========================================================

import clickhouse_connect
from datetime import datetime
from croniter import croniter
from dagster import job, op, resource, sensor, RunRequest, Definitions
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
import os, sys, time, uuid


# ================= CONFIG =================

CH_HOST = os.getenv("CH_HOST", "localhost")
CH_PORT = int(os.getenv("CH_PORT", 8123))
CH_USER = os.getenv("CH_USER", "default")
CH_PASSWORD = os.getenv("CH_PASSWORD", "")
CH_DB = os.getenv("CH_DB", "default")

CLUSTER = "y_credal_cluster"
MAX_CONCURRENCY = 4


# ================= LOGGING =================

logger.remove()
logger.add(sys.stdout, level="INFO")
logger.add("datamart.log", rotation="500 MB", retention="14 days")


# ================= CLICKHOUSE RESOURCE =================

@resource
def clickhouse():

    client = clickhouse_connect.get_client(
        host=CH_HOST,
        port=CH_PORT,
        username=CH_USER,
        password=CH_PASSWORD,
        database=CH_DB,
    )

    client.command(f"""
    CREATE TABLE IF NOT EXISTS datamart_config ON CLUSTER {CLUSTER}
    (
        datamart_name String,
        target_table String,
        query String,
        schedule_cron String,

        depends_on Array(String),

        chunk_columns String,  -- NEW: "col1,col2,col3"
        chunk_size UInt64,

        concurrency UInt8 DEFAULT 1,

        enabled UInt8 DEFAULT 1,
        last_run DateTime DEFAULT toDateTime(0),

        created_at DateTime DEFAULT now()
    )
    ENGINE = ReplicatedMergeTree('/clickhouse/tables/{{shard}}/datamart_config', '{{replica}}')
    ORDER BY datamart_name
    """)

    client.command(f"""
    CREATE TABLE IF NOT EXISTS datamart_log ON CLUSTER {CLUSTER}
    (
        run_id String,
        datamart_name String,
        target_table String,
        status String,
        message String,

        start_time DateTime,
        end_time DateTime,
        duration_sec Float64,

        chunk_count UInt32,
        query String,

        created_at DateTime DEFAULT now()
    )
    ENGINE = ReplicatedMergeTree('/clickhouse/tables/{{shard}}/datamart_log', '{{replica}}')
    ORDER BY (datamart_name, start_time)
    """)

    return client


# ================= DAG CHECK =================

def deps_ok(client, deps):
    if not deps:
        return True

    for d in deps:
        r = client.query(f"""
            SELECT max(end_time)
            FROM datamart_log
            WHERE datamart_name='{d}'
              AND status='success'
        """).result_rows[0][0]

        if r is None:
            return False

    return True


# ================= COMPOSITE CHUNKING =================

def parse_columns(chunk_columns: str):
    return [c.strip() for c in chunk_columns.split(",") if c.strip()]


def build_where_clause(cols, start_tuple, end_tuple):

    conditions = []

    for i, col in enumerate(cols):
        conditions.append(
            f"{col} >= {start_tuple[i]} AND {col} < {end_tuple[i]}"
        )

    return " AND ".join(conditions)


def get_bounds(client, query, cols):

    select_cols = ",".join([f"min({c}), max({c})" for c in cols])

    res = client.query(f"SELECT {select_cols} FROM ({query})").result_rows[0]

    bounds = []
    for i in range(0, len(res), 2):
        bounds.append((res[i], res[i+1]))

    return bounds


def generate_chunks(bounds, size):

    # simple grid chunking (safe baseline)
    # could be improved later with histogram partitioning

    chunks = []

    def recurse(level, current_start, path):

        if level == len(bounds):
            chunks.append(tuple(path))
            return

        start, end = bounds[level]

        cur = start
        while cur <= end:
            nxt = cur + size
            recurse(level + 1, cur, path + [cur])
            cur = nxt

    recurse(0, 0, [])
    return chunks


# ================= EXECUTION =================

def run_chunk(client, table, query, cols, start, end):

    where = build_where_clause(cols, start, end)

    client.command(f"""
        INSERT INTO {table}
        SELECT * FROM ({query})
        WHERE {where}
    """)


def run_composite_chunks(client, table, query, cols, size, concurrency):

    bounds = get_bounds(client, query, cols)
    chunks = generate_chunks(bounds, size)

    concurrency = min(concurrency or 1, MAX_CONCURRENCY)

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [
            ex.submit(run_chunk, client, table, query, cols, s, e)
            for s, e in chunks
        ]
        for f in as_completed(futures):
            f.result()

    return len(chunks)


# ================= OP =================

@op(required_resource_keys={"clickhouse"}, config_schema=dict)
def run_datamart(context):

    cfg = context.op_config
    client = context.resources.clickhouse

    run_id = context.run_id
    name = cfg["datamart_name"]
    target = cfg["target_table"]

    tmp = f"{target}__tmp_{uuid.uuid4().hex[:6]}"

    start = datetime.utcnow()
    t0 = time.time()

    status = "success"
    message = "success"
    chunk_count = 0

    try:

        client.command(f"CREATE TABLE {tmp} AS {target}")

        if cfg.get("chunk_columns"):

            cols = parse_columns(cfg["chunk_columns"])

            chunk_count = run_composite_chunks(
                client,
                tmp,
                cfg["query"],
                cols,
                cfg["chunk_size"],
                cfg.get("concurrency", 1)
            )

        else:

            client.command(f"INSERT INTO {tmp} {cfg['query']}")
            chunk_count = 1

        client.command(f"""
            RENAME TABLE {target} TO {target}__old,
                         {tmp} TO {target}
        """)

        client.command(f"DROP TABLE {target}__old")

    except Exception as e:

        status = "failed"
        message = str(e).replace("'", " ")

        client.command(f"DROP TABLE IF EXISTS {tmp}")
        raise

    finally:

        duration = round(time.time() - t0, 2)
        end = datetime.utcnow()

        client.command(f"""
        INSERT INTO datamart_log VALUES (
            '{run_id}',
            '{name}',
            '{target}',
            '{status}',
            '{message}',
            toDateTime('{start}'),
            toDateTime('{end}'),
            {duration},
            {chunk_count},
            $$ {cfg["query"]} $$,
            now()
        )
        """)


# ================= JOB =================

@job(resource_defs={"clickhouse": clickhouse})
def datamart_job():
    run_datamart()


# ================= SENSOR =================

@sensor(job=datamart_job, minimum_interval_seconds=60)
def datamart_sensor(context):

    client = clickhouse_connect.get_client(
        host=CH_HOST,
        port=CH_PORT,
        username=CH_USER,
        password=CH_PASSWORD,
        database=CH_DB,
    )

    rows = client.query("SELECT * FROM datamart_config WHERE enabled=1").result_rows
    now = datetime.utcnow()

    for r in rows:

        name, target, query, cron, deps, chunk_cols, size, conc, _, last_run, _ = r

        if not deps_ok(client, deps):
            continue

        itr = croniter(cron, last_run)

        if itr.get_next(datetime) <= now:

            yield RunRequest(
                run_key=f"{name}_{now}",
                run_config={"ops":{"run_datamart":{"config":{
                    "datamart_name":name,
                    "target_table":target,
                    "query":query,
                    "chunk_columns":chunk_cols,
                    "chunk_size":size,
                    "concurrency":conc
                }}}}
            )


defs = Definitions(
    jobs=[datamart_job],
    sensors=[datamart_sensor],
    resources={"clickhouse": clickhouse},
)






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