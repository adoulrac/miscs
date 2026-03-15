“””
terminal_front.py — Terminal K8s Streamlit
Appelle le backend FastAPI, ne touche jamais au cluster directement.
Tout exec passe par le streaming SSE (POST /exec).
“””

import streamlit as st
import requests

# ── Config ────────────────────────────────────────────────────────────────────

API_BASE = “http://localhost:8000”   # URL de ton backend FastAPI

st.set_page_config(
page_title=“K8s Terminal”,
page_icon=“⬛”,
layout=“wide”,
initial_sidebar_state=“expanded”,
)

# ── CSS terminal ──────────────────────────────────────────────────────────────

st.markdown(”””

<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');

  .stApp { background-color: #0d1117; color: #c9d1d9; font-family: 'JetBrains Mono', monospace; }

  section[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
  section[data-testid="stSidebar"] * { font-family: 'JetBrains Mono', monospace !important; color: #8b949e !important; }
  section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] h1,
  section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: #58a6ff !important; }

  .terminal-window { background:#010409; border:1px solid #30363d; border-radius:8px;
                     overflow:hidden; box-shadow:0 8px 32px rgba(0,0,0,.6); margin-bottom:12px; }
  .terminal-titlebar { background:#161b22; padding:8px 14px; display:flex; align-items:center;
                       gap:8px; border-bottom:1px solid #30363d; }
  .dot { width:12px; height:12px; border-radius:50%; display:inline-block; }
  .dot-r{background:#ff5f57} .dot-y{background:#febc2e} .dot-g{background:#28c840}
  .terminal-title { flex:1; text-align:center; font-size:12px; color:#8b949e; font-family:'JetBrains Mono',monospace; }
  .terminal-body { padding:14px 16px; min-height:380px; max-height:460px; overflow-y:auto;
                   font-family:'JetBrains Mono',monospace; font-size:13px; line-height:1.65;
                   background:#010409; white-space:pre-wrap; word-break:break-all; }
  .terminal-body::-webkit-scrollbar{width:5px}
  .terminal-body::-webkit-scrollbar-thumb{background:#30363d;border-radius:3px}

  .line-output{color:#c9d1d9} .line-error{color:#f85149} .line-info{color:#3fb950}
  .line-cmd{color:#79c0ff}    .line-system{color:#6e7681;font-style:italic}
  .line-warning{color:#d29922}

  .badge{display:inline-block;padding:3px 10px;border-radius:4px;font-size:11px;margin-right:4px}
  .badge-ns {background:#1f3a5f;color:#58a6ff;border:1px solid #1f6feb}
  .badge-pod{background:#1a3a2a;color:#3fb950;border:1px solid #238636}
  .badge-ctr{background:#3a2a1a;color:#d29922;border:1px solid #9e6a03}

  .stTextInput label{display:none!important}
  .stTextInput>div>div>input{
    background:transparent!important; border:none!important; box-shadow:none!important;
    color:#e6edf3!important; font-family:'JetBrains Mono',monospace!important;
    font-size:13px!important; caret-color:#3fb950;
  }
  .stButton>button{
    background:#21262d; color:#c9d1d9; border:1px solid #30363d;
    font-family:'JetBrains Mono',monospace; font-size:12px;
    padding:5px 12px; border-radius:5px; transition:all .15s;
  }
  .stButton>button:hover{background:#30363d;border-color:#58a6ff;color:#58a6ff}

  .status-bar{background:#161b22;border:1px solid #30363d;border-radius:5px;
              padding:6px 14px;font-size:11px;color:#6e7681;font-family:'JetBrains Mono',monospace;
              display:flex;gap:16px;}
  .status-ok{color:#3fb950} .status-err{color:#f85149}
</style>

“””, unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────

def init():
defaults = {
“history”: [],          # (type, text)
“namespaces”: [],
“pods”: [],
“containers”: [],
“selected_ns”: None,
“selected_pod”: None,
“selected_container”: None,
“connected”: False,
“cmd_count”: 0,
“err_count”: 0,
}
for k, v in defaults.items():
if k not in st.session_state:
st.session_state[k] = v

init()

# ── Helpers API ───────────────────────────────────────────────────────────────

def api_get(path: str, **params):
“”“GET vers le backend FastAPI, retourne le JSON ou None.”””
try:
r = requests.get(f”{API_BASE}{path}”, params=params, timeout=10)
r.raise_for_status()
return r.json()
except requests.exceptions.ConnectionError:
add_line(“error”, f”[API] Impossible de joindre le backend ({API_BASE})”)
return None
except requests.exceptions.HTTPError as e:
add_line(“error”, f”[API] Erreur HTTP {e.response.status_code}”)
return None

def api_stream(payload: dict):
“””
POST streaming vers /exec — utilisé pour toutes les commandes.
Retourne un générateur de chunks : stdout:<ligne> / stderr:<ligne> / **END**
“””
try:
with requests.post(
f”{API_BASE}/exec”, json=payload, stream=True, timeout=60
) as resp:
resp.raise_for_status()
for raw in resp.iter_lines():
if raw:
line = raw.decode(“utf-8”) if isinstance(raw, bytes) else raw
# SSE format : “data: …”
if line.startswith(“data: “):
yield line[6:]   # strip “data: “
except Exception as e:
yield f”stderr:{e}”
yield “**END**”

def api_log_stream(namespace, pod, container, tail=100, follow=False):
“”“GET streaming vers /logs/{ns}/{pod}.”””
try:
params = {“container”: container, “tail_lines”: tail, “follow”: str(follow).lower()}
with requests.get(
f”{API_BASE}/logs/{namespace}/{pod}”,
params=params, stream=True, timeout=60
) as resp:
resp.raise_for_status()
for raw in resp.iter_lines():
if raw:
line = raw.decode(“utf-8”) if isinstance(raw, bytes) else raw
if line.startswith(“data: “):
yield line[6:]
except Exception as e:
yield f”[ERROR] {e}”

# ── Terminal helpers ──────────────────────────────────────────────────────────

def add_line(ltype: str, text: str):
st.session_state.history.append((ltype, text))

def render_terminal():
html = “”
for ltype, text in st.session_state.history:
esc = text.replace(”&”, “&”).replace(”<”, “<”).replace(”>”, “>”)
html += f’<div class="line-{ltype}">{esc}</div>\n’
return html

def is_streaming_command(cmd: str) -> bool:
“”“Détermine si la commande mérite un streaming (longue durée).”””
streaming_keywords = [“tail -f”, “watch “, “top”, “htop”, “ping”, “npm run”, “python -m”]
return any(kw in cmd for kw in streaming_keywords)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
st.markdown(”## ⬛ K8s Terminal”)
st.caption(f”Backend : `{API_BASE}`”)
st.divider()

```
# Charger les namespaces
if not st.session_state.namespaces:
    if st.button("🔌 Charger les namespaces", use_container_width=True):
        data = api_get("/namespaces")
        if data:
            st.session_state.namespaces = data["namespaces"]
            st.rerun()
else:
    st.markdown("### Namespace")
    ns = st.selectbox("ns", st.session_state.namespaces,
                      key="ns_select", label_visibility="collapsed")

    # Recharge les pods si namespace change
    if ns != st.session_state.selected_ns:
        st.session_state.selected_ns = ns
        data = api_get(f"/namespaces/{ns}/pods")
        if data:
            st.session_state.pods = [p["name"] for p in data["pods"]]
            # Stocke aussi les containers par pod
            st.session_state._pods_meta = {p["name"]: p["containers"] for p in data["pods"]}
        st.session_state.selected_pod = None
        st.session_state.connected = False
        st.rerun()

    if st.session_state.pods:
        st.markdown("### Pod")
        pod = st.selectbox("pod", st.session_state.pods,
                           key="pod_select", label_visibility="collapsed")

        if pod != st.session_state.selected_pod:
            st.session_state.selected_pod = pod
            pods_meta = st.session_state.get("_pods_meta", {})
            st.session_state.containers = pods_meta.get(pod, [])
            st.session_state.selected_container = None
            st.session_state.connected = False

        if st.session_state.containers:
            st.markdown("### Container")
            ctr = st.selectbox("ctr", st.session_state.containers,
                               key="ctr_select", label_visibility="collapsed")
            st.session_state.selected_container = ctr

            st.divider()
            if st.button("▶ exec -it", use_container_width=True):
                st.session_state.connected = True
                add_line("system", f"Connecting to {pod}/{ctr} in namespace {ns}...")
                add_line("info",   "✓ Session ouverte — tapez vos commandes")
                add_line("system", "─" * 52)
                st.rerun()

            if st.session_state.connected:
                if st.button("✕ Déconnecter", use_container_width=True):
                    st.session_state.connected = False
                    add_line("system", "─" * 52)
                    add_line("system", "Session terminée.")
                    st.rerun()

            # Bouton logs rapides
            st.divider()
            if st.button("📋 Voir les logs (100 lignes)", use_container_width=True):
                add_line("cmd", f"# logs {pod}/{ctr} (tail 100)")
                for line in api_log_stream(ns, pod, ctr, tail=100, follow=False):
                    add_line("output", line)
                st.session_state.cmd_count += 1
                st.rerun()

st.divider()
if st.button("🗑 Clear terminal", use_container_width=True):
    st.session_state.history = []
    st.session_state.cmd_count = 0
    st.session_state.err_count = 0
    st.rerun()
```

# ── Main area ─────────────────────────────────────────────────────────────────

ns  = st.session_state.selected_ns
pod = st.session_state.selected_pod
ctr = st.session_state.selected_container

# Badges ou hint

if st.session_state.connected and pod:
st.markdown(
f’<span class="badge badge-ns">ns: {ns}</span>’
f’<span class="badge badge-pod">pod: {pod}</span>’
f’<span class="badge badge-ctr">ctr: {ctr}</span>’,
unsafe_allow_html=True,
)
else:
st.markdown(’<span style="color:#484f58;font-family:JetBrains Mono,monospace;font-size:13px;">’
‘Chargez les namespaces → sélectionnez un pod → ▶ exec -it</span>’,
unsafe_allow_html=True)

st.markdown(””)

# Fenêtre terminal

tab_label = f”{pod} — {ctr}” if st.session_state.connected else “terminal”
body_html  = render_terminal()

st.markdown(f”””

<div class="terminal-window">
  <div class="terminal-titlebar">
    <span class="dot dot-r"></span><span class="dot dot-y"></span><span class="dot dot-g"></span>
    <span class="terminal-title">{tab_label}</span>
  </div>
  <div class="terminal-body">
    {body_html if body_html else '<span class="line-system">Prêt.</span>'}
  </div>
</div>
""", unsafe_allow_html=True)

# ── Zone de saisie ────────────────────────────────────────────────────────────

if st.session_state.connected:
prompt = f”{ctr}@{pod[:24]}:~$”
col_p, col_i, col_r = st.columns([2, 8, 1.2])

```
with col_p:
    st.markdown(
        f'<div style="padding-top:8px;font-family:JetBrains Mono,monospace;'
        f'font-size:13px;color:#3fb950;">{prompt}</div>',
        unsafe_allow_html=True,
    )
with col_i:
    cmd = st.text_input("cmd", key="cmd_field",
                        placeholder="ls -la /app", label_visibility="collapsed")
with col_r:
    run = st.button("⏎ Run", use_container_width=True)

if run and cmd.strip():
    add_line("cmd", f"{prompt} {cmd}")
    st.session_state.cmd_count += 1

    payload = {"namespace": ns, "pod": pod, "container": ctr, "command": cmd}
    for chunk in api_stream(payload):
        if chunk == "__END__":
            break
        if chunk.startswith("stderr:"):
            add_line("error", chunk[7:])
            st.session_state.err_count += 1
        elif chunk.startswith("stdout:"):
            add_line("output", chunk[7:])
        else:
            add_line("output", chunk)

    st.rerun()
```

else:
st.markdown(
‘<div style="background:#010409;border:1px solid #21262d;border-radius:0 0 8px 8px;'
'padding:10px 16px;font-family:JetBrains Mono,monospace;font-size:13px;color:#484f58;">’
‘$ — sélectionnez un pod pour démarrer une session</div>’,
unsafe_allow_html=True,
)

# ── Barre de statut ───────────────────────────────────────────────────────────

st.markdown(””)
c = st.session_state.cmd_count
e = st.session_state.err_count
k8s_ok   = ‘<span class="status-ok">● k8s</span>’
sess_st  = ‘<span class="status-ok">● session active</span>’ if st.session_state.connected   
else ‘<span style="color:#484f58">○ idle</span>’
err_col  = “status-err” if e > 0 else “”
err_st   = f’<span class="{err_col}">{e} err</span>’

st.markdown(f”””

<div class="status-bar">
  {k8s_ok} &nbsp;|&nbsp; {sess_st} &nbsp;|&nbsp;
  <span>{c} cmd(s)</span> &nbsp;|&nbsp; {err_st}
</div>
""", unsafe_allow_html=True)