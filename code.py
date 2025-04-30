

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