
from loader import run_pipeline, suggest_clickhouse_schema
import pandas as pd

def get_data():
    return pd.DataFrame({
        "user_id": [1, 2, 3],
        "region": ["US", "EU", "US"],
        "amount": [100.5, 200.0, 150.75],
        "active": [True, False, True],
        "signup_date": pd.to_datetime(["2023-01-01", "2023-02-15", "2023-03-10"])
    })

# Option 1: Let ClickHouse infer types from data
# run_pipeline(get_data, table_name="users")

# Option 2: Use recommendation
df = get_data()
recommended_schema = suggest_clickhouse_schema(df)
run_pipeline(get_data, table_name="users", schema=recommended_schema)

# Option 3: Provide manual schema
# custom_schema = {
#     "user_id": "UInt32",
#     "region": "LowCardinality(String) CODEC(ZSTD)",
#     "amount": "Float64",
#     "active": "UInt8",
#     "signup_date": "DateTime"
# }
# run_pipeline(get_data, table_name="users", schema=custom_schema)





import pandas as pd
import clickhouse_connect
import uuid
from datetime import datetime
from typing import Callable, Optional
from logger import setup_logger
from config import CLICKHOUSE_CONFIG

logger = setup_logger("ClickHouseLoader")


def create_client():
    logger.info("Connecting to ClickHouse...")
    try:
        client = clickhouse_connect.get_client(
            host=CLICKHOUSE_CONFIG["host"],
            port=CLICKHOUSE_CONFIG["port"],
            username=CLICKHOUSE_CONFIG["username"],
            password=CLICKHOUSE_CONFIG["password"],
            database=CLICKHOUSE_CONFIG["database"]
        )
        logger.info("Connected to ClickHouse.")
        return client
    except Exception as e:
        logger.exception("Failed to connect to ClickHouse.")
        raise e


def ensure_main_table_exists(client, table_name: str, schema: dict):
    logger.info(f"Ensuring table '{table_name}' exists...")
    try:
        column_defs = [f"{col} {dtype}" for col, dtype in schema.items()]
        column_defs.append("load_id UUID")
        column_defs.append("load_timestamp DateTime")
        ddl = f'''
        CREATE TABLE IF NOT EXISTS {CLICKHOUSE_CONFIG["database"]}.{table_name} (
            {', '.join(column_defs)}
        ) ENGINE = MergeTree()
        ORDER BY tuple()
        '''
        client.command(ddl)
        logger.info(f"Table '{table_name}' is ready.")
    except Exception as e:
        logger.exception(f"Failed to ensure table '{table_name}'.")
        raise e


def ensure_log_table_exists(client):
    logger.info("Ensuring log table exists...")
    try:
        ddl = f'''
        CREATE TABLE IF NOT EXISTS {CLICKHOUSE_CONFIG["database"]}.load_log (
            load_id UUID,
            load_timestamp DateTime,
            table_name String,
            row_count UInt32,
            status String,
            error_message String
        ) ENGINE = MergeTree()
        ORDER BY load_timestamp
        '''
        client.command(ddl)
        logger.info("Log table is ready.")
    except Exception as e:
        logger.exception("Failed to ensure log table.")
        raise e


def log_load(client, load_id, timestamp, table_name, row_count, status, error_message=""):
    try:
        log_df = pd.DataFrame([{
            "load_id": load_id,
            "load_timestamp": timestamp,
            "table_name": table_name,
            "row_count": row_count,
            "status": status,
            "error_message": error_message
        }])
        client.insert_df("load_log", log_df)
        logger.info(f"Logged load event for table '{table_name}': {status}")
    except Exception as e:
        logger.error("⚠️ Failed to log load event.")
        logger.exception(e)


def load_dataframe_to_clickhouse(df: pd.DataFrame, client, table_name: str, load_id: str, timestamp: datetime):
    if df.empty:
        logger.warning("DataFrame is empty. Skipping insert.")
        return 0

    try:
        df["load_id"] = load_id
        df["load_timestamp"] = timestamp

        logger.info(f"Inserting {len(df)} rows into '{table_name}'...")
        client.insert_df(table_name, df)
        logger.info("Insert complete.")
        return len(df)
    except Exception as e:
        logger.exception(f"Failed to insert into '{table_name}'.")
        raise e


def infer_schema_from_df(df: pd.DataFrame) -> dict:
    schema = {}
    for col, dtype in df.dtypes.items():
        if pd.api.types.is_integer_dtype(dtype):
            schema[col] = "Int64"
        elif pd.api.types.is_float_dtype(dtype):
            schema[col] = "Float64"
        elif pd.api.types.is_bool_dtype(dtype):
            schema[col] = "UInt8"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            schema[col] = "DateTime"
        else:
            schema[col] = "String"
    return schema


def suggest_clickhouse_schema(df: pd.DataFrame) -> dict:
    schema = {}
    for col in df.columns:
        series = df[col].dropna()
        dtype = series.dtype

        if series.empty:
            schema[col] = "String"
            continue

        unique_ratio = series.nunique() / len(series)

        if pd.api.types.is_string_dtype(dtype):
            if unique_ratio < 0.5:
                schema[col] = "LowCardinality(String)"
            else:
                schema[col] = "String"

        elif pd.api.types.is_integer_dtype(dtype):
            min_val, max_val = series.min(), series.max()
            if min_val >= 0:
                if max_val < 256:
                    schema[col] = "UInt8"
                elif max_val < 65536:
                    schema[col] = "UInt16"
                elif max_val < 2**32:
                    schema[col] = "UInt32"
                else:
                    schema[col] = "UInt64"
            else:
                if -128 <= min_val <= 127:
                    schema[col] = "Int8"
                elif -32768 <= min_val <= 32767:
                    schema[col] = "Int16"
                elif -2**31 <= min_val <= 2**31 - 1:
                    schema[col] = "Int32"
                else:
                    schema[col] = "Int64"

        elif pd.api.types.is_float_dtype(dtype):
            schema[col] = "Float32" if series.std() < 1e-3 else "Float64"

        elif pd.api.types.is_bool_dtype(dtype):
            schema[col] = "UInt8"

        elif pd.api.types.is_datetime64_any_dtype(dtype):
            schema[col] = "DateTime"

        else:
            schema[col] = "String"

    return schema


def run_pipeline(
    df_supplier: Callable[[], pd.DataFrame],
    table_name: str,
    schema: Optional[dict] = None
):
    load_id = str(uuid.uuid4())
    timestamp = datetime.utcnow()

    client = create_client()
    ensure_log_table_exists(client)

    try:
        df = df_supplier()
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Data supplier must return a pandas DataFrame.")

        if schema is None:
            schema = infer_schema_from_df(df)

        ensure_main_table_exists(client, table_name, schema)
        row_count = load_dataframe_to_clickhouse(df, client, table_name, load_id, timestamp)
        log_load(client, load_id, timestamp, table_name, row_count, status="success")
    except Exception as e:
        logger.error("Pipeline failed.")
        log_load(client, load_id, timestamp, table_name, 0, status="failed", error_message=str(e))
        raise e



import logging

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.hasHandlers():
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger




import os
from dotenv import load_dotenv

load_dotenv()

CLICKHOUSE_CONFIG = {
    "host": os.getenv("CLICKHOUSE_HOST", "localhost"),
    "port": int(os.getenv("CLICKHOUSE_PORT", 8123)),
    "username": os.getenv("CLICKHOUSE_USER", "default"),
    "password": os.getenv("CLICKHOUSE_PASSWORD", ""),
    "database": os.getenv("CLICKHOUSE_DATABASE", "default")
}






Properties props = new Properties();
props.put("bootstrap.servers", "your-broker:9092");
props.put("key.serializer", "io.confluent.kafka.serializers.KafkaAvroSerializer");
props.put("value.serializer", "io.confluent.kafka.serializers.KafkaAvroSerializer");

props.put("schema.registry.url", "https://your-schema-registry:8081");

// Authentification SASL/PLAIN
props.put("security.protocol", "SASL_SSL");  // ou SASL_PLAINTEXT
props.put("sasl.mechanism", "PLAIN");
props.put("sasl.jaas.config",
    "org.apache.kafka.common.security.plain.PlainLoginModule required " +
    "username=\"SVC_XXX\" " +
    "password=\"your_password\";");




from confluent_kafka import Consumer, TopicPartition
import time

def reset_offsets_at_timestamp(consumer, topic, timestamp_ms):
    # Subscribe to topic
    consumer.subscribe([topic])

    # Force a poll to ensure the assignment is complete
    while not consumer.assignment():
        consumer.poll(1.0)  # Trigger the rebalance
    
    # Get current assignments
    partitions = consumer.assignment()

    # Map each partition to the desired timestamp
    timestamps = [TopicPartition(p.topic, p.partition, timestamp_ms) for p in partitions]

    # Look up the offset for the given timestamp
    offsets = consumer.offsets_for_times(timestamps, timeout=10)

    # Seek to the desired offset for each partition
    for tp in offsets:
        if tp.offset != -1:
            consumer.seek(tp)
        else:
            print(f"No offset available for partition {tp.partition} at timestamp {timestamp_ms}")

# Example usage
conf = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'my-group',
    'auto.offset.reset': 'earliest',
    'enable.auto.commit': False
}
consumer = Consumer(conf)

timestamp_ms = int(time.time() - 3600) * 1000  # 1 hour ago
reset_offsets_at_timestamp(consumer, 'my-topic', timestamp_ms)

# Continue consuming
while True:
    msg = consumer.poll(1.0)
    if msg is None:
        continue
    if msg.error():
        print(f"Error: {msg.error()}")
    else:
        print(f"Received message: {msg.value().decode('utf-8')}")

consumer.close(




from confluent_kafka.avro import AvroConsumer
from confluent_kafka.avro.serializer import SerializerError
from confluent_kafka import TopicPartition
from datetime import datetime
import time


def create_avro_consumer(broker, schema_registry_url, group_id="my-avro-consumer"):
    consumer_config = {
        'bootstrap.servers': broker,
        'group.id': group_id,
        'auto.offset.reset': 'none',  # on veut contrôler le point de départ
        'schema.registry.url': schema_registry_url
    }

    return AvroConsumer(consumer_config)


def assign_consumer_to_timestamp(consumer, topic, timestamp_dt):
    # 1. Convertir datetime -> timestamp en millisecondes
    timestamp_ms = int(time.mktime(timestamp_dt.timetuple()) * 1000)

    # 2. Obtenir les partitions du topic
    metadata = consumer.list_topics(topic, timeout=10)
    partitions = metadata.topics[topic].partitions

    topic_partitions = [TopicPartition(topic, p, timestamp_ms) for p in partitions]

    # 3. Chercher les offsets pour ce timestamp
    offsets = consumer.offsets_for_times(topic_partitions, timeout=10)

    # 4. Assigner les partitions avec l'offset correspondant
    valid_offsets = [tp for tp in offsets if tp.offset != -1]

    if not valid_offsets:
        raise ValueError("Aucun offset trouvé pour le timestamp fourni.")

    consumer.assign(valid_offsets)
    print(f"Assigned to offsets: {valid_offsets}")


def consume_messages(consumer):
    print("Consuming messages. Press Ctrl+C to stop.")
    try:
        while True:
            try:
                msg = consumer.poll(1.0)
                if msg is None:
                    continue

                if msg.error():
                    print(f"Error: {msg.error()}")
                    continue

                print("✔ Offset:", msg.offset())
                print("✔ Key:", msg.key())
                print("✔ Value:", msg.value())
                print("---")

            except SerializerError as e:
                print("❌ Deserialization error:", e)
                continue
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        consumer.close()





from confluent_kafka.avro import AvroConsumer
from confluent_kafka.avro.serializer import SerializerError


def create_avro_consumer(broker, schema_registry_url, topic, group_id="my-avro-consumer"):
    consumer_config = {
        'bootstrap.servers': broker,
        'group.id': group_id,
        'auto.offset.reset': 'earliest',
        'schema.registry.url': schema_registry_url
    }

    consumer = AvroConsumer(consumer_config)
    consumer.subscribe([topic])
    return consumer


def consume_messages(consumer):
    print("Consuming messages. Press Ctrl+C to stop.")
    try:
        while True:
            try:
                msg = consumer.poll(1.0)
                if msg is None:
                    continue

                if msg.error():
                    print(f"Error: {msg.error()}")
                    continue

                print("✔ Message Key:", msg.key())
                print("✔ Message Value:", msg.value())
                print("---")

            except SerializerError as e:
                print("❌ Deserialization error:", e)
                continue
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        consumer.close()


if __name__ == "__main__":
    broker = "localhost:9092"  # 🔁 Remplace par l'adresse de ton broker
    schema_registry_url = "http://localhost:8081"  # 🔁 URL de ton Schema Registry
    topic = "ton-topic"  # 🔁 Remplace par le nom de ton topic

    consumer = create_avro_consumer(broker, schema_registry_url, topic)
    consume_messages(consumer)



import requests
from datetime import datetime, timedelta, timezone

# === CONFIGURATION ===
DAGSTER_GRAPHQL_URL = "http://localhost:3000/graphql"  # Modifiez si besoin
AUTH_TOKEN = None  # Remplacez par un token si nécessaire ("Bearer abc123...")

headers = {
    "Content-Type": "application/json",
}
if AUTH_TOKEN:
    headers["Authorization"] = f"Bearer {AUTH_TOKEN}"

# === QUERY GraphQL ===
query = """
query GetTodayRuns($since: Float!) {
  runs(filter: { createdAfter: $since }) {
    runId
    status
    pipelineName
    startTime
    endTime
  }
}
"""

# Calcul de minuit aujourd'hui en UTC
now = datetime.now(timezone.utc)
midnight = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
midnight_timestamp = midnight.timestamp()

response = requests.post(
    DAGSTER_GRAPHQL_URL,
    json={"query": query, "variables": {"since": midnight_timestamp}},
    headers=headers
)

if response.status_code != 200:
    print("Erreur HTTP :", response.status_code)
    print(response.text)
    exit(1)

data = response.json()

if "errors" in data:
    print("Erreur GraphQL :", data["errors"])
    exit(1)

runs = data["data"]["runs"]

print(f"\n🎯 Travaux Dagster lancés aujourd'hui ({len(runs)}):\n")

for run in runs:
    name = run["pipelineName"]
    run_id = run["runId"]
    status = run["status"]
    start = run.get("startTime")
    end = run.get("endTime")

    if start:
        start_dt = datetime.fromtimestamp(start, tz=timezone.utc)
        duration = (datetime.now(timezone.utc) - start_dt).total_seconds() / 60
    else:
        duration = 0

    if status == "STARTED":
        if duration > 60:
            state = "⏱️ En cours depuis plus d'une heure"
        else:
            state = "🟡 En cours"
    elif status == "SUCCESS":
        state = "✅ Succès"
    elif status == "FAILURE":
        state = "❌ Échec"
    else:
        state = f"🔘 Statut : {status}"

    print(f"- {name} ({run_id[:8]}) → {state}")




import asyncio
import websockets
import json

async def rmds_session():
    uri = "wss://<rmds-endpoint>"
    async with websockets.connect(uri) as ws:
        # Send login
        await ws.send(json.dumps({
            "ID": 1,
            "Domain": "Login",
            "Key": {"Name": "YourUserName"}
        }))
        
        # Listen or interact as needed
        try:
            while True:
                message = await ws.recv()
                print(message)
                # Handle messages...

        except asyncio.CancelledError:
            print("Task cancelled, cleaning up...")
            await clean_close(ws)
            raise
        
        except Exception as e:
            print(f"Error: {e}")
            await clean_close(ws)

async def clean_close(ws):
    try:
        # Optionally send RMDS logout / close message
        await ws.send(json.dumps({
            "ID": 1,
            "Type": "Close"
        }))
    except Exception as e:
        print(f"Error sending RMDS close: {e}")

    # Initiate WebSocket close handshake
    await ws.close()
    await ws.wait_closed()
    print("WebSocket closed cleanly.")

# To stop the connection cleanly from outside:
# task = asyncio.create_task(rmds_session())
# task.cancel()
# await task

asyncio.run(rmds_session())



import panel as pn
import pandas as pd
from datetime import datetime, timedelta

def get_big_df():
    now = datetime.now()
    expiration_minutes = 60 * 24  # 1 day

    if 'df' not in pn.state.cache or 'df_time' not in pn.state.cache:
        print("Loading df first time...")
        pn.state.cache['df'] = pd.read_csv('myfile.csv')
        pn.state.cache['df_time'] = now
    elif (now - pn.state.cache['df_time']) > timedelta(minutes=expiration_minutes):
        print("Reloading expired df...")
        pn.state.cache['df'] = pd.read_csv('myfile.csv')
        pn.state.cache['df_time'] = now

    return pn.state.cache['df']



import time
from kombu.exceptions import OperationalError
from tasks import app

def safe_add_consumer(queue_name, retries=5, base_delay=1):
    for attempt in range(retries):
        try:
            app.control.add_consumer(queue_name)
            print(f"Successfully added queue: {queue_name}")
            return
        except OperationalError as e:
            wait = base_delay * (2 ** attempt)
            print(f"Attempt {attempt+1}: Redis error: {e} — retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed to add consumer after {retries} retries.")


import redis
import time
from datetime import datetime

def get_latest_timestamps(stream_names, redis_host="localhost", redis_port=6379):
    """
    Fetches the latest event timestamp from a list of Redis streams and returns them as datetime objects.

    :param stream_names: List of Redis stream names
    :param redis_host: Redis server host (default: "localhost")
    :param redis_port: Redis server port (default: 6379)
    :return: Dictionary with stream names as keys and (latest event datetime, current datetime)
    """
    r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
    current_timestamp = int(time.time() * 1000)  # Current timestamp in milliseconds
    current_datetime = datetime.utcfromtimestamp(current_timestamp / 1000)  # Convert to UTC datetime

    result = {}
    for stream in stream_names:
        stream_info = r.xinfo_stream(stream, full=False)
        last_entry_id = stream_info.get("last-entry", None)

        if last_entry_id:
            latest_event_timestamp = int(last_entry_id.split('-')[0])  # Extract milliseconds part
            latest_event_datetime = datetime.utcfromtimestamp(latest_event_timestamp / 1000)  # Convert to UTC datetime
        else:
            latest_event_datetime = None  # No events in stream

        result[stream] = {
            "latest_event_datetime": latest_event_datetime,
            "current_datetime": current_datetime
        }

    return result






import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import datetime
import json
import os
import time

# --------------- Configuration ---------------
API_BASE_URL = "https://your-api-endpoint.com"  # Replace with your API URL
THRESHOLDS_FILE = "pipeline_thresholds.json"
REFRESH_INTERVAL = 15  # Auto-refresh every 15 seconds

# --------------- Helper Functions ---------------
def fetch_pipelines():
    """Fetch pipeline data from the API."""
    try:
        response = requests.get(f"{API_BASE_URL}/pipelines")
        return response.json()
    except Exception:
        return []

def fetch_pipeline_errors():
    """Fetch errors occurring in the last 5 minutes."""
    try:
        response = requests.get(f"{API_BASE_URL}/errors?last_seconds=300")  # Last 5 min
        return response.json()
    except Exception:
        return []

def load_thresholds():
    """Load custom thresholds from a JSON file."""
    if os.path.exists(THRESHOLDS_FILE):
        with open(THRESHOLDS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_thresholds(thresholds):
    """Save custom thresholds to a JSON file."""
    with open(THRESHOLDS_FILE, "w") as f:
        json.dump(thresholds, f, indent=4)

def get_thresholds(pipeline_name):
    """Get warning & critical thresholds for a pipeline."""
    return thresholds.get(pipeline_name, {"warning": 5, "critical": 10})

# --------------- Load Data ---------------
st.set_page_config(page_title="Pipeline Monitoring", layout="wide")

# Auto-refresh every 15 seconds
st.experimental_set_query_params(auto_refresh=int(time.time()))  
time.sleep(REFRESH_INTERVAL)  
st.experimental_rerun()  

# Load pipeline thresholds
thresholds = load_thresholds()

# Fetch pipeline data
pipelines = fetch_pipelines()
df = pd.DataFrame(pipelines)

# Convert data types
df["last_refresh"] = pd.to_datetime(df["last_refresh"])
df["latency"] = df["latency"].astype(float)
df["errors"] = df["errors"].astype(int)

# --------------- TABS FOR DASHBOARD & SETTINGS ---------------
tab1, tab2 = st.tabs(["📊 Dashboard", "⚙️ Settings"])

# ======================= 📊 MAIN DASHBOARD =======================
with tab1:
    st.title("🚀 Pipeline Monitoring Dashboard")
    st.write(f"Last Updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # **KPI Metrics**
    total_pipelines = len(df)
    failed_pipelines = df["errors"].sum()
    high_latency_pipelines = len(df[df["latency"] > df["latency"].map(lambda x: get_thresholds(x)['critical'])])
    average_latency = df["latency"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Pipelines", total_pipelines)
    col2.metric("❌ Total Errors", failed_pipelines)
    col3.metric("⚠️ High Latency Pipelines", high_latency_pipelines)
    col4.metric("⏳ Average Latency (s)", f"{average_latency:.2f}")

    # **Live Status Cards**
    st.subheader("🔍 Live Pipeline Status")
    cols = st.columns(5)
    for i, row in df.iterrows():
        pipeline_name = row["name"]
        thresholds_for_pipeline = get_thresholds(pipeline_name)
        color = "red" if row["errors"] > 0 else "green"
        error_alert = "❌" if row["errors"] > 0 else "✅"

        with cols[i % 5]:
            if row["errors"] > 0:
                if st.button(f"{error_alert} {pipeline_name} ({row['errors']} Errors)", key=f"error_{pipeline_name}"):
                    errors = fetch_pipeline_errors()
                    df_errors = pd.DataFrame(errors)
                    df_errors = df_errors[df_errors["pipeline"] == pipeline_name]
                    st.write(f"**Error Logs for {pipeline_name}**")
                    st.dataframe(df_errors)
            else:
                st.markdown(
                    f"""
                    <div style="border: 2px solid {color}; padding: 10px; border-radius: 10px;">
                        <h4>{pipeline_name}</h4>
                        <p>🕒 Last Refresh: {row['last_refresh']}</p>
                        <p>⚡ Latency: {row['latency']}s</p>
                        <p>✅ No Errors</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # **Latency Trends**
    st.subheader("📈 Latency Trends Over Time")
    fig = px.line(df, x="last_refresh", y="latency", color="name", title="Latency Trends")
    st.plotly_chart(fig, use_container_width=True)

    # **Latency Heatmap**
    st.subheader("🔥 Latency Heatmap")
    df_heatmap = df.pivot(index="last_refresh", columns="name", values="latency")
    fig_heatmap = px.imshow(df_heatmap, labels={"color": "Latency (s)"}, title="Latency Heatmap")
    st.plotly_chart(fig_heatmap, use_container_width=True)

# ======================= ⚙️ SETTINGS PAGE =======================
with tab2:
    st.title("⚙️ Configure Pipeline Thresholds")
    st.write("Adjust the warning and critical latency thresholds for each pipeline.")

    for pipeline_name in df["name"].unique():
        st.subheader(f"🔧 {pipeline_name}")

        current_thresholds = get_thresholds(pipeline_name)

        new_warning = st.slider(
            f"⚠️ Warning Threshold for {pipeline_name}",
            min_value=1, max_value=20,
            value=current_thresholds["warning"],
            key=f"{pipeline_name}_warning"
        )
        new_critical = st.slider(
            f"🚨 Critical Threshold for {pipeline_name}",
            min_value=5, max_value=50,
            value=current_thresholds["critical"],
            key=f"{pipeline_name}_critical"
        )

        thresholds[pipeline_name] = {"warning": new_warning, "critical": new_critical}

    if st.button("💾 Save Thresholds"):
        save_thresholds(thresholds)
        st.success("Thresholds saved successfully! Refresh the dashboard to apply changes.")












import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import datetime
import json
import os

# --------------- Configuration ---------------
API_BASE_URL = "https://your-api-endpoint.com"  # Replace with your API URL
THRESHOLDS_FILE = "pipeline_thresholds.json"

# --------------- Helper Functions ---------------
def fetch_pipelines():
    """Fetch pipeline data from the API."""
    try:
        response = requests.get(f"{API_BASE_URL}/pipelines")
        return response.json()
    except Exception as e:
        return []

def fetch_pipeline_errors():
    """Fetch errors occurring in the last ERROR_CHECK_WINDOW seconds."""
    try:
        response = requests.get(f"{API_BASE_URL}/errors?last_seconds=60")
        return response.json()
    except Exception as e:
        return []

def load_thresholds():
    """Load custom thresholds from a JSON file."""
    if os.path.exists(THRESHOLDS_FILE):
        with open(THRESHOLDS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_thresholds(thresholds):
    """Save custom thresholds to a JSON file."""
    with open(THRESHOLDS_FILE, "w") as f:
        json.dump(thresholds, f, indent=4)

def get_thresholds(pipeline_name):
    """Get warning & critical thresholds for a pipeline."""
    return thresholds.get(pipeline_name, {"warning": 5, "critical": 10})

def get_status_color(latency, warning_threshold, critical_threshold):
    """Return color based on pipeline-specific thresholds."""
    if latency > critical_threshold:
        return "red"
    elif latency > warning_threshold:
        return "orange"
    return "green"

# --------------- Load Data ---------------
st.set_page_config(page_title="Pipeline Monitoring", layout="wide")

# Load pipeline thresholds
thresholds = load_thresholds()

# Fetch pipeline data
pipelines = fetch_pipelines()
df = pd.DataFrame(pipelines)

# Convert to correct data types
df["last_refresh"] = pd.to_datetime(df["last_refresh"])
df["latency"] = df["latency"].astype(float)
df["errors"] = df["errors"].astype(int)

# --------------- TABS FOR DASHBOARD & SETTINGS ---------------
tab1, tab2 = st.tabs(["📊 Dashboard", "⚙️ Settings"])

# ======================= 📊 MAIN DASHBOARD =======================
with tab1:
    st.title("🚀 Pipeline Monitoring Dashboard")
    st.write(f"Last Updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Compute Summary Stats
    total_pipelines = len(df)
    failed_pipelines = len(df[df["errors"] > 0])
    high_latency_pipelines = len(df[df["latency"] > df["latency"].map(lambda x: get_thresholds(x)['critical'])])
    average_latency = df["latency"].mean()

    # **KPI Metrics**
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Pipelines", total_pipelines)
    col2.metric("❌ Failed Pipelines", failed_pipelines)
    col3.metric("⚠️ High Latency Pipelines", high_latency_pipelines)
    col4.metric("⏳ Average Latency (s)", f"{average_latency:.2f}")

    # **Live Status Cards**
    st.subheader("🔍 Live Pipeline Status")
    cols = st.columns(5)
    for i, row in df.iterrows():
        pipeline_name = row["name"]
        thresholds_for_pipeline = get_thresholds(pipeline_name)
        color = get_status_color(row["latency"], thresholds_for_pipeline["warning"], thresholds_for_pipeline["critical"])

        with cols[i % 5]:
            st.markdown(
                f"""
                <div style="border: 2px solid {color}; padding: 10px; border-radius: 10px;">
                    <h4>{pipeline_name}</h4>
                    <p>🕒 Last Refresh: {row['last_refresh']}</p>
                    <p>⚡ Latency: {row['latency']}s</p>
                    <p>❌ Errors: {row['errors']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # **Latency Trends**
    st.subheader("📈 Latency Trends Over Time")
    fig = px.line(df, x="last_refresh", y="latency", color="name", title="Latency Trends")
    st.plotly_chart(fig, use_container_width=True)

# ======================= ⚙️ SETTINGS PAGE =======================
with tab2:
    st.title("⚙️ Configure Pipeline Thresholds")
    st.write("Adjust the warning and critical latency thresholds for each pipeline.")

    # Editable Thresholds
    for pipeline_name in df["name"].unique():
        st.subheader(f"🔧 {pipeline_name}")

        # Get current thresholds
        current_thresholds = get_thresholds(pipeline_name)

        # Create sliders for editing
        new_warning = st.slider(
            f"⚠️ Warning Threshold for {pipeline_name}",
            min_value=1, max_value=20,
            value=current_thresholds["warning"],
            key=f"{pipeline_name}_warning"
        )
        new_critical = st.slider(
            f"🚨 Critical Threshold for {pipeline_name}",
            min_value=5, max_value=50,
            value=current_thresholds["critical"],
            key=f"{pipeline_name}_critical"
        )

        # Update local dictionary
        thresholds[pipeline_name] = {"warning": new_warning, "critical": new_critical}

    # Save Button
    if st.button("💾 Save Thresholds"):
        save_thresholds(thresholds)
        st.success("Thresholds saved successfully! Refresh the dashboard to apply changes.")



from celery import Celery

# Celery configuration
BROKER_URL = "redis://localhost:6379/0"  # Change if using another broker
app = Celery("monitor", broker=BROKER_URL)

# Queue name of your worker
QUEUE_NAME = "my_queue"  # Change this to match your worker's queue name

def get_failed_tasks():
    """
    Get failed tasks from Celery and print them.
    """
    inspect = app.control.inspect()
    failed_tasks = inspect.failed()

    if not failed_tasks:
        print("✅ No failed tasks found.")
        return

    for worker, tasks in failed_tasks.items():
        if QUEUE_NAME in worker:  # Check if worker is handling the specified queue
            for task_id, task_info in tasks.items():
                print(f"⚠️ Failed Task on {worker}: Task ID {task_id}, Exception: {task_info['exception']}")

if __name__ == "__main__":
    get_failed_tasks()












from kubernetes import client, config
import re

# Muatkan konfigurasi Kubernetes (gunakan in-cluster jika berjalan dalam pod)
try:
    config.load_incluster_config()  # Jika berjalan dalam Kubernetes
except config.ConfigException:
    config.load_kube_config()  # Jika berjalan di luar Kubernetes (guna kubeconfig)

# Buat API client
v1 = client.CoreV1Api()

def get_pod_logs(namespace, deployment_name):
    """
    Dapatkan log daripada pod dalam deployment tertentu.
    """
    try:
        # Senaraikan semua pod dalam namespace yang diberikan
        pods = v1.list_namespaced_pod(namespace)

        for pod in pods.items:
            # Semak sama ada nama pod mengandungi nama deployment
            if deployment_name in pod.metadata.name:
                print(f"Fetching logs for pod: {pod.metadata.name}")

                # Ambil log daripada pod
                logs = v1.read_namespaced_pod_log(name=pod.metadata.name, namespace=namespace)
                return logs

    except Exception as e:
        print(f"Error retrieving logs: {e}")

    return None  # Tiada pod yang sepadan

def check_errors_in_logs(logs):
    """
    Periksa jika terdapat ralat dalam log pod.
    """
    error_patterns = [
        r"error", r"exception", r"failed", r"traceback"
    ]
    
    for line in logs.split("\n"):
        if any(re.search(pattern, line, re.IGNORECASE) for pattern in error_patterns):
            print(f"Error detected: {line}")
            return True

    return False

def main():
    """
    Fungsi utama untuk mendapatkan log dan mengesan sebarang ralat.
    """
    namespace = "your-namespace"  # Gantikan dengan namespace anda
    deployment_name = "your-deployment"  # Gantikan dengan nama deployment anda

    logs = get_pod_logs(namespace, deployment_name)

    if logs:
        if check_errors_in_logs(logs):
            print("Errors detected in logs!")
        else:
            print("No errors found.")
    else:
        print("No logs retrieved.")

if __name__ == "__main__":
    main()
















import pandas as pd
import ast

# Sample DataFrame with NaN and incorrect values
data = {
    "id": [1, 2, 3, 4, 5],
    "dict_column": ['{"key1": "A", "key2": "B"}', '{"key1": "C", "key2": "D"}', None, 'hello', '123'],  # None (NaN), invalid string, number
    "filter_key": ["key1", "key2", "key1", "key1", "key2"]
}

df = pd.DataFrame(data)

# Improved extraction logic with error handling
def safe_extract(row):
    value = row["dict_column"]
    key = row["filter_key"]
    
    if pd.isna(value):  # Skip NaN
        return None
    
    try:
        parsed_dict = ast.literal_eval(value) if isinstance(value, str) else value  # Parse only if string
        return parsed_dict.get(key, None) if isinstance(parsed_dict, dict) else None  # Ensure it's a dict
    except (ValueError, SyntaxError):  # Catch invalid literals
        return None

df["filtered_value"] = df.apply(safe_extract, axis=1)

print(df)



import pandas as pd
import ast

# Sample DataFrame with NaN and incorrect values
data = {
    "id": [1, 2, 3, 4, 5],
    "dict_column": ['{"key1": "A", "key2": "B"}', '{"key1": "C", "key2": "D"}', None, 'hello', '123'],
    "filter_key": ["key1", "key2", "key1", "key1", "key2"]
}

df = pd.DataFrame(data)

# Compact extraction logic
df["filtered_value"] = df.apply(
    lambda row: (d := (ast.literal_eval(row["dict_column"]) if isinstance(row["dict_column"], str) else None)) and d.get(row["filter_key"])
    if isinstance(d, dict) else None,
    axis=1
)

print(df)






import pandas as pd

data = {
    "name": [
        "deltaFET / open interest", "delta / market space cap", "delta / volume", 
        "delta / volume 30 days", "delta / VWAP volume", "delta / auction volume", 
        "delta", "notification", "session buys / volume", "session sales / volume", 
        "close buys / auction volume", "close sales / auction volume", 
        "tot buys / open interest", "tot buys / market cap", "session buys / volume 30 days", 
        "tot sales / open interest", "tot sales / market cap", "session sales / volume 30 days"
    ],
    "value": [
        0.01, 0.01, 0.2, 0.2, 0.2, 0.15, 1_000_000, 0, 0.1, 0.1, 
        0.1, 0.1, 0.01, 0.01, 0.1, 0.01, 0.01, 0.1
    ],
    "text value": [
        "", "", "", "", "", "", "", "touched.*", "", "", "", "", 
        "", "", "", "", "", ""
    ],
    "financial category MRX": [
        "", "", "", "", "", "(not in) future.bond index", "", "", "", "", 
        "", "", "", "", "", "", "", ""
    ]
}

df = pd.DataFrame(data)
print(df)





from collections import deque, defaultdict
import fnmatch

def extract_related_data(data: dict, start_key: str, exclude_patterns: list = None) -> dict:
    """
    Extrait un dictionnaire filtré en conservant la structure d'origine et en récupérant
    toutes les entrées liées selon un système de préfixes (avant le point '.').
    
    Optimisations :
    - Pré-indexe les clés selon leur préfixe pour accélérer les recherches.
    - Utilise une deque pour améliorer la gestion des clés à traiter.
    - Exclut les clés qui correspondent à l'un des modèles de filtre dans la liste.

    :param data: Dictionnaire source.
    :param start_key: Clé de départ pour l'extraction.
    :param exclude_patterns: Liste des patterns des clés à exclure (optionnel).
    :return: Dictionnaire filtré avec toutes les dépendances résolues.
    """
    
    # Étape 1 : Construire un index des clés par préfixe
    prefix_map = defaultdict(set)
    for key in data:
        prefix = key.split('.')[0]
        prefix_map[prefix].add(key)

    # Étape 2 : Parcourir les clés liées
    if start_key not in data:
        return {}

    extracted = {}
    seen_keys = set()
    keys_to_process = deque([start_key])

    # Vérifier si la clé correspond à l'un des patterns d'exclusion
    def is_excluded(key):
        if exclude_patterns:
            return any(fnmatch.fnmatch(key, pattern) for pattern in exclude_patterns)
        return False

    while keys_to_process:
        key = keys_to_process.popleft()
        
        # Vérifier si la clé est exclue selon l'un des patterns
        if is_excluded(key):
            continue

        if key in seen_keys:
            continue
        seen_keys.add(key)

        if key in data:
            value = data[key]
            extracted[key] = value  # Conserver la structure

            # Si la valeur est une chaîne, vérifier si elle correspond à un préfixe existant
            if isinstance(value, str) and value in prefix_map:
                keys_to_process.extend(prefix_map[value])

            # Si la valeur est une liste, traiter chaque élément
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item in prefix_map:
                        keys_to_process.extend(prefix_map[item])

    return extracted

# Exemple d'utilisation
data = {
    "user.name": "Alice",
    "user.age": 30,
    "user.city": "Paris",
    "product.name": "Laptop",
    "ignore.this": "should be excluded",
    "user.preferences.color": "blue",
    "ignore.other": "should also be excluded"
}

# Extraction des données liées pour "user" tout en excluant les clés qui correspondent aux modèles
exclude_patterns = ["ignore.*", "product.*"]
result = extract_related_data(data, "user", exclude_patterns)
print(result)




from collections import deque, defaultdict
import fnmatch

def extract_related_data(data: dict, start_key: str, exclude_pattern: str = None) -> dict:
    """
    Extrait un dictionnaire filtré en conservant la structure d'origine et en récupérant
    toutes les entrées liées selon un système de préfixes (avant le point '.').
    
    Optimisations :
    - Pré-indexe les clés selon leur préfixe pour accélérer les recherches.
    - Utilise une deque pour améliorer la gestion des clés à traiter.
    - Exclut les clés qui correspondent au pattern de filtre.

    :param data: Dictionnaire source.
    :param start_key: Clé de départ pour l'extraction.
    :param exclude_pattern: Le pattern des clés à exclure (optionnel).
    :return: Dictionnaire filtré avec toutes les dépendances résolues.
    """
    
    # Étape 1 : Construire un index des clés par préfixe
    prefix_map = defaultdict(set)
    for key in data:
        prefix = key.split('.')[0]
        prefix_map[prefix].add(key)

    # Étape 2 : Parcourir les clés liées
    if start_key not in data:
        return {}

    extracted = {}
    seen_keys = set()
    keys_to_process = deque([start_key])

    while keys_to_process:
        key = keys_to_process.popleft()
        
        # Vérifier si la clé est exclue selon le pattern
        if exclude_pattern and fnmatch.fnmatch(key, exclude_pattern):
            continue

        if key in seen_keys:
            continue
        seen_keys.add(key)

        if key in data:
            value = data[key]
            extracted[key] = value  # Conserver la structure

            # Si la valeur est une chaîne, vérifier si elle correspond à un préfixe existant
            if isinstance(value, str) and value in prefix_map:
                keys_to_process.extend(prefix_map[value])

            # Si la valeur est une liste, traiter chaque élément
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item in prefix_map:
                        keys_to_process.extend(prefix_map[item])

    return extracted

# Exemple d'utilisation
data = {
    "user.name": "Alice",
    "user.age": 30,
    "user.city": "Paris",
    "product.name": "Laptop",
    "ignore.this": "should be excluded",
    "user.preferences.color": "blue"
}

# Extraction des données liées pour "user" tout en excluant les clés qui commencent par "ignore"
result = extract_related_data(data, "user", "ignore.*")
print(result)





import fnmatch

def extract_related_data_optimized(d: dict, key_to_extract: str, exclude_pattern: str = None) -> dict:
    """
    Optimized version of extract_related_data function with exclusion of keys based on patterns.
    
    :param d: The dictionary to extract data from
    :param key_to_extract: The key whose related data should be extracted
    :param exclude_pattern: The pattern to match keys that should be excluded (default is None)
    :return: A new dictionary with extracted related data and excluded keys
    """
    result = {}

    for key, value in d.items():
        # Skip keys that match the exclude pattern
        if exclude_pattern and fnmatch.fnmatch(key, exclude_pattern):
            continue
        
        # Extract the relevant key-value pair
        if key_to_extract in key:
            result[key] = value

        # Recursively handle nested dictionaries and lists
        if isinstance(value, dict):
            nested_result = extract_related_data_optimized(value, key_to_extract, exclude_pattern)
            if nested_result:
                result[key] = nested_result

        elif isinstance(value, list):
            new_list = []
            for item in value:
                if isinstance(item, dict):
                    new_item = extract_related_data_optimized(item, key_to_extract, exclude_pattern)
                    if new_item:
                        new_list.append(new_item)
                else:
                    new_list.append(item)
            if new_list:
                result[key] = new_list
        else:
            result[key] = value

    return result

# Example of use
data = {
    "user.name": "Alice",
    "user.age": 30,
    "user.city": "Paris",
    "product.name": "Laptop",
    "ignore.this": "should be excluded",
    "user.preferences.color": "blue"
}

# Extract related data for "user" and exclude any keys that match "ignore.*"
result = extract_related_data_optimized(data, "user", "ignore.*")
print(result)




import fnmatch

def remove_prefix_and_exclude_keys(d: dict, prefix: str, exclude_pattern: str = None) -> dict:
    """
    Remove a prefix from all keys in the dictionary and optionally exclude keys matching a pattern.
    
    :param d: The dictionary to process
    :param prefix: The prefix to remove from the keys
    :param exclude_pattern: The pattern to match keys that should be excluded (default is None)
    :return: A new dictionary with modified keys
    """
    result = {}

    for key, value in d.items():
        # Skip keys that match the exclude pattern
        if exclude_pattern and fnmatch.fnmatch(key, exclude_pattern):
            continue
        
        # Remove the prefix if the key starts with it
        new_key = key[len(prefix):] if key.startswith(prefix) else key
        
        # Recursively process the value if it's a dictionary or list
        if isinstance(value, dict):
            result[new_key] = remove_prefix_and_exclude_keys(value, prefix, exclude_pattern)
        elif isinstance(value, list):
            result[new_key] = [remove_prefix_and_exclude_keys(v, prefix, exclude_pattern) if isinstance(v, dict) else v for v in value]
        else:
            result[new_key] = value
    
    return result

# Example of use
data = {
    "user.name": "Alice",
    "user.age": 30,
    "user.city": "Paris",
    "product.name": "Laptop",
    "ignore.this": "should be excluded"
}

# Remove 'user.' prefix and exclude keys that match 'ignore.*' pattern
result = remove_prefix_and_exclude_keys(data, "user.", "ignore.*")
print(result)




from collections import deque, defaultdict

def extract_related_data(data: dict, start_key: str) -> dict:
    """
    Extrait un dictionnaire filtré en conservant la structure d'origine et en récupérant
    toutes les entrées liées selon un système de préfixes (avant le point '.').
    
    Optimisations :
    - Pré-indexe les clés selon leur préfixe pour accélérer les recherches.
    - Utilise une deque pour améliorer la gestion des clés à traiter.

    :param data: Dictionnaire source.
    :param start_key: Clé de départ pour l'extraction.
    :return: Dictionnaire filtré avec toutes les dépendances résolues.
    """

    # Étape 1 : Construire un index des clés par préfixe
    prefix_map = defaultdict(set)
    for key in data:
        prefix = key.split('.')[0]
        prefix_map[prefix].add(key)

    # Étape 2 : Parcourir les clés liées
    if start_key not in data:
        return {}

    extracted = {}
    seen_keys = set()
    keys_to_process = deque([start_key])

    while keys_to_process:
        key = keys_to_process.popleft()
        if key in seen_keys:
            continue
        seen_keys.add(key)

        if key in data:
            value = data[key]
            extracted[key] = value  # Conserver la structure

            # Si la valeur est une chaîne, vérifier si elle correspond à un préfixe existant
            if isinstance(value, str) and value in prefix_map:
                keys_to_process.extend(prefix_map[value])

            # Si la valeur est une liste, traiter chaque élément
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item in prefix_map:
                        keys_to_process.extend(prefix_map[item])

    return extracted



from collections.abc import Iterable, Mapping

def extract_related_data(data: dict, start_key: str) -> dict:
    """
    Extracts a filtered dictionary that maintains the exact input structure.
    It resolves subdependencies by looking at values and matching keys based on their prefix (before the dot).

    :param data: The input dictionary.
    :param start_key: The key to start extraction from.
    :return: A filtered dictionary with all related key-value pairs.
    """
    def get_related_keys(value):
        """Finds all keys where the prefix before the dot matches the given value."""
        return {k for k in data if k.split('.')[0] == value}

    def resolve_keys(keys_to_process, seen_keys):
        """Recursively processes keys, adding related keys dynamically."""
        while keys_to_process:
            key = keys_to_process.pop()
            if key in seen_keys:
                continue
            seen_keys.add(key)

            if key in data:
                value = data[key]
                extracted[key] = value  # Keep the exact structure

                # Check if value references another key prefix
                if isinstance(value, str):
                    keys_to_process.update(get_related_keys(value))
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str):
                            keys_to_process.update(get_related_keys(item))

    if start_key not in data:
        return {}

    extracted = {}
    resolve_keys({start_key}, set())
    return extracted





from collections.abc import Iterable, Mapping

def extract_related_dict(data: dict, key: str) -> dict:
    """
    Extracts a filtered dictionary that includes only the relevant key and all its related sub-objects.
    Maintains the original dictionary structure.

    :param data: The input dictionary.
    :param key: The key to start extraction from.
    :return: A filtered dictionary containing the key and its related sub-objects.
    """
    def collect_keys(value, collected):
        """Recursively collect all keys that are referenced in the dictionary."""
        if isinstance(value, str) and value in data and value not in collected:
            collected.add(value)
            collect_keys(data[value], collected)
        elif isinstance(value, list):
            for item in value:
                collect_keys(item, collected)
        elif isinstance(value, dict):
            for v in value.values():
                collect_keys(v, collected)

    if key not in data:
        return {}

    # Step 1: Find all related keys
    related_keys = {key}
    collect_keys(data[key], related_keys)

    # Step 2: Filter the original dictionary to keep only related keys
    filtered_dict = {k: v for k, v in data.items() if k in related_keys}

    return filtered_dict




from pydantic import BaseModel
from typing import Dict, Any
import pandas as pd

class MyDataModel(BaseModel):
    data: Dict[str, Any]  # Accepts a dictionary with any key-value pairs

def extract_prefixed_keys(model: MyDataModel, prefix: str) -> pd.DataFrame:
    # Extract relevant keys and values
    filtered_items = {k[len(prefix):]: v for k, v in model.data.items() if k.startswith(prefix)}
    
    # Convert to DataFrame
    df = pd.DataFrame([filtered_items])  # Wrap in list to create single-row DataFrame
    
    return df

# Example dictionary
large_dict = {
    "price_AAPL": 150,
    "price_GOOG": 2800,
    "price_MSFT": 299,
    "volume_AAPL": 1000,
    "volume_GOOG": 2000
}

# Convert to Pydantic model
model = MyDataModel(data=large_dict)

# Extract DataFrame for "price_" prefix
df_prices = extract_prefixed_keys(model, "price_")
print(df_prices)






import ast

def parse_dictionary(input_dict):
    def parse_value(value):
        """Convert string representations of lists into actual lists."""
        if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass
        return value

    def build_nested_structure(keys, value, structure):
        """Recursively build nested dictionaries from a list of keys."""
        if len(keys) == 1:
            structure[keys[0]] = value
        else:
            if keys[0] not in structure:
                structure[keys[0]] = {}
            build_nested_structure(keys[1:], value, structure[keys[0]])

    def resolve_prefixes(parsed_dict):
        """Resolve prefixes by checking if a value (or list item) is a prefix of another key."""
        final_dict = {}
        for key, value in parsed_dict.items():
            if isinstance(value, list):
                # Check if any item in the list is a prefix for another key
                for item in value:
                    if isinstance(item, str):
                        # Check if this item is a prefix for another key
                        for other_key in parsed_dict:
                            if other_key != key and other_key.startswith(item + '.'):
                                # Create a nested structure for the prefix
                                if item not in final_dict:
                                    final_dict[item] = {}
                                build_nested_structure(
                                    other_key[len(item) + 1:].split('.'),
                                    parsed_dict[other_key],
                                    final_dict[item]
                                )
            elif isinstance(value, str):
                # Check if this value is a prefix for another key
                for other_key in parsed_dict:
                    if other_key != key and other_key.startswith(value + '.'):
                        # Create a nested structure for the prefix
                        if value not in final_dict:
                            final_dict[value] = {}
                        build_nested_structure(
                            other_key[len(value) + 1:].split('.'),
                            parsed_dict[other_key],
                            final_dict[value]
                        )
            # Add the current key-value pair to the final dictionary
            if key not in final_dict:
                build_nested_structure(key.split('.'), value, final_dict)
        return final_dict

    # First pass: Parse values and build the initial dictionary
    parsed_dict = {}
    for key, value in input_dict.items():
        parsed_dict[key] = parse_value(value)

    # Second pass: Resolve prefixes and build the final nested structure
    final_dict = resolve_prefixes(parsed_dict)
    return final_dict

# Example usage:
input_dict = {
    'user.name': 'John Doe',
    'user.age': '30',
    'user.hobbies': '["reading", "traveling", "coding"]',
    'address.city': 'New York',
    'address.zip': '10001',
    'reading.property': 'value for reading',
    'coding.difficulty': 'high',
    'traveling.destination': 'Japan'
}

parsed_dict = parse_dictionary(input_dict)
print(parsed_dict)