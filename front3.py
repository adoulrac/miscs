“””
terminal_front.py — Terminal K8s Streamlit
Session shell persistante : cd fonctionne, le CWD est affiché dans le prompt.
Entrée sur le champ de commande = exécution immédiate (pas besoin de cliquer Run).
“””

import streamlit as st
import requests

API_BASE = “http://localhost:8000”

st.set_page_config(
page_title=“K8s Terminal”,
page_icon=“⬛”,
layout=“wide”,
initial_sidebar_state=“expanded”,
)

# ── CSS ────────────────────────────────────────────────────────────────────────

st.markdown(”””

<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');

  .stApp { background:#0d1117; color:#c9d1d9; font-family:'JetBrains Mono',monospace; }

  section[data-testid="stSidebar"] { background:#161b22; border-right:1px solid #30363d; }
  section[data-testid="stSidebar"] * { font-family:'JetBrains Mono',monospace !important; color:#8b949e !important; }
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] h1,
  section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3 { color:#58a6ff !important; }

  .terminal-window { background:#010409; border:1px solid #30363d; border-radius:8px;
                     overflow:hidden; box-shadow:0 8px 32px rgba(0,0,0,.6); margin-bottom:8px; }
  .terminal-titlebar { background:#161b22; padding:8px 14px; display:flex; align-items:center;
                       gap:8px; border-bottom:1px solid #30363d; }
  .dot { width:12px; height:12px; border-radius:50%; display:inline-block; }
  .dot-r{background:#ff5f57} .dot-y{background:#febc2e} .dot-g{background:#28c840}
  .terminal-title { flex:1; text-align:center; font-size:12px; color:#8b949e; font-family:'JetBrains Mono',monospace; }

  .terminal-body { padding:14px 16px; min-height:400px; max-height:480px; overflow-y:auto;
                   font-family:'JetBrains Mono',monospace; font-size:13px; line-height:1.65;
                   background:#010409; white-space:pre-wrap; word-break:break-all; }
  .terminal-body::-webkit-scrollbar{width:5px}
  .terminal-body::-webkit-scrollbar-thumb{background:#30363d;border-radius:3px}

  .line-output {color:#c9d1d9}
  .line-stderr {color:#f85149}
  .line-info   {color:#3fb950}
  .line-cmd    {color:#79c0ff; font-weight:bold;}
  .line-system {color:#6e7681; font-style:italic}

  .badge{display:inline-block;padding:3px 10px;border-radius:4px;font-size:11px;margin-right:4px}
  .badge-ns {background:#1f3a5f;color:#58a6ff;border:1px solid #1f6feb}
  .badge-pod{background:#1a3a2a;color:#3fb950;border:1px solid #238636}
  .badge-ctr{background:#3a2a1a;color:#d29922;border:1px solid #9e6a03}

  .stTextInput label{display:none !important}
  .stTextInput>div>div>input {
    background:transparent !important; border:none !important; box-shadow:none !important;
    color:#e6edf3 !important; font-family:'JetBrains Mono',monospace !important;
    font-size:13px !important; caret-color:#3fb950;
  }
  .stButton>button {
    background:#21262d; color:#c9d1d9; border:1px solid #30363d;
    font-family:'JetBrains Mono',monospace; font-size:12px;
    padding:5px 12px; border-radius:5px; transition:all .15s;
  }
  .stButton>button:hover{background:#30363d;border-color:#58a6ff;color:#58a6ff}

  .status-bar{background:#161b22;border:1px solid #30363d;border-radius:5px;
              padding:6px 14px;font-size:11px;color:#6e7681;
              font-family:'JetBrains Mono',monospace;display:flex;gap:16px;}
  .status-ok{color:#3fb950} .status-err{color:#f85149}
</style>

“””, unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────

def init():
defaults = {
“history”:    [],       # (type, text)
“namespaces”: [],
“pods”:       [],
“containers”: [],
“_pods_meta”: {},
“selected_ns”:        None,
“selected_pod”:       None,
“selected_container”: None,
“session_id”:  None,    # ID de la session shell persistante
“cwd”:         “~”,     # CWD courant dans le pod
“cmd_count”:   0,
“err_count”:   0,
“last_cmd”:    “”,      # pour détecter Entrée sans re-exec
}
for k, v in defaults.items():
if k not in st.session_state:
st.session_state[k] = v

init()

# ── Helpers API ────────────────────────────────────────────────────────────────

def api_get(path: str, **params):
try:
r = requests.get(f”{API_BASE}{path}”, params=params, timeout=10)
r.raise_for_status()
return r.json()
except requests.exceptions.ConnectionError:
add_line(“system”, f”[API] Backend inaccessible ({API_BASE})”)
return None
except requests.exceptions.HTTPError as e:
add_line(“stderr”, f”[API] HTTP {e.response.status_code}”)
return None

def api_post(path: str, payload: dict):
try:
r = requests.post(f”{API_BASE}{path}”, json=payload, timeout=15)
r.raise_for_status()
return r.json()
except requests.exceptions.ConnectionError:
add_line(“system”, f”[API] Backend inaccessible ({API_BASE})”)
return None
except requests.exceptions.HTTPError as e:
add_line(“stderr”, f”[API] HTTP {e.response.status_code}: {e.response.text[:100]}”)
return None

def api_delete(path: str):
try:
requests.delete(f”{API_BASE}{path}”, timeout=5)
except Exception:
pass

def api_exec_stream(session_id: str, command: str):
“””
POST /exec en streaming SSE.
Yields (type, text) : stdout | stderr | cwd | **END**
“””
try:
with requests.post(
f”{API_BASE}/exec”,
json={“session_id”: session_id, “command”: command},
stream=True, timeout=60,
) as resp:
resp.raise_for_status()
for raw in resp.iter_lines():
if not raw:
continue
line = raw.decode(“utf-8”) if isinstance(raw, bytes) else raw
if not line.startswith(“data: “):
continue
payload = line[6:]               # strip “data: “
if payload == “**END**”:
yield (”**END**”, “”)
return
if “:” in payload:
kind, _, text = payload.partition(”:”)
yield (kind, text)
except Exception as e:
yield (“stderr”, str(e))
yield (”**END**”, “”)

def api_log_stream(namespace, pod, container, tail=100):
try:
params = {“container”: container, “tail_lines”: tail, “follow”: “false”}
with requests.get(
f”{API_BASE}/logs/{namespace}/{pod}”,
params=params, stream=True, timeout=30,
) as resp:
resp.raise_for_status()
for raw in resp.iter_lines():
if raw:
line = raw.decode(“utf-8”) if isinstance(raw, bytes) else raw
if line.startswith(“data: “):
yield line[6:]
except Exception as e:
yield f”[ERROR] {e}”

# ── Terminal helpers ───────────────────────────────────────────────────────────

def add_line(ltype: str, text: str):
st.session_state.history.append((ltype, text))

def render_terminal():
html = “”
for ltype, text in st.session_state.history:
esc = text.replace(”&”, “&”).replace(”<”, “<”).replace(”>”, “>”)
html += f’<div class="line-{ltype}">{esc}</div>\n’
return html

def prompt_str():
“”“Construit le prompt affiché, ex: api-gateway@pod:/app$”””
ctr = st.session_state.selected_container or “”
pod = (st.session_state.selected_pod or “”)[:20]
cwd = st.session_state.cwd or “~”
return f”{ctr}@{pod}:{cwd}$”

def run_command(cmd: str):
“”“Exécute la commande via l’API et met à jour l’historique + CWD.”””
sid = st.session_state.session_id
prompt = prompt_str()
add_line(“cmd”, f”{prompt} {cmd}”)
st.session_state.cmd_count += 1

```
for kind, text in api_exec_stream(sid, cmd):
    if kind == "__END__":
        break
    elif kind == "cwd":
        st.session_state.cwd = text       # ← met à jour le prompt
    elif kind == "stdout":
        add_line("output", text)
    elif kind == "stderr":
        add_line("stderr", text)
        st.session_state.err_count += 1
```

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
st.markdown(”## ⬛ K8s Terminal”)
st.caption(f”`{API_BASE}`”)
st.divider()

```
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

    if ns != st.session_state.selected_ns:
        st.session_state.selected_ns = ns
        data = api_get(f"/namespaces/{ns}/pods")
        if data:
            st.session_state.pods = [p["name"] for p in data["pods"]]
            st.session_state._pods_meta = {p["name"]: p["containers"] for p in data["pods"]}
        st.session_state.selected_pod = None
        st.session_state.selected_container = None
        st.session_state.session_id = None
        st.rerun()

    if st.session_state.pods:
        st.markdown("### Pod")
        pod = st.selectbox("pod", st.session_state.pods,
                           key="pod_select", label_visibility="collapsed")

        if pod != st.session_state.selected_pod:
            st.session_state.selected_pod = pod
            st.session_state.containers = st.session_state._pods_meta.get(pod, [])
            st.session_state.selected_container = None
            st.session_state.session_id = None

        if st.session_state.containers:
            st.markdown("### Container")
            ctr = st.selectbox("ctr", st.session_state.containers,
                               key="ctr_select", label_visibility="collapsed")
            st.session_state.selected_container = ctr

            st.divider()

            # Bouton de connexion / reconnexion
            btn_label = "🔄 Reconnecter" if st.session_state.session_id else "▶ Ouvrir le terminal"
            if st.button(btn_label, use_container_width=True):
                # Ferme l'ancienne session si elle existe
                if st.session_state.session_id:
                    api_delete(f"/session/{st.session_state.session_id}")

                data = api_post("/session/open", {
                    "namespace": ns, "pod": pod, "container": ctr
                })
                if data:
                    st.session_state.session_id = data["session_id"]
                    st.session_state.cwd = data.get("cwd", "~")
                    add_line("system", f"Connecting to {pod}/{ctr} — namespace: {ns}")
                    add_line("info",   f"✓ Shell ouvert (session: {data['session_id'][:8]}…)")
                    add_line("system", "─" * 52)
                st.rerun()

            if st.session_state.session_id:
                if st.button("✕ Déconnecter", use_container_width=True):
                    api_delete(f"/session/{st.session_state.session_id}")
                    st.session_state.session_id = None
                    st.session_state.cwd = "~"
                    add_line("system", "─" * 52)
                    add_line("system", "Session terminée.")
                    st.rerun()

            st.divider()
            if st.button("📋 Logs (100 lignes)", use_container_width=True):
                add_line("cmd", f"# kubectl logs {pod} -c {ctr} --tail=100")
                for line in api_log_stream(ns, pod, ctr, tail=100):
                    add_line("output", line)
                st.rerun()

st.divider()
if st.button("🗑 Clear", use_container_width=True):
    st.session_state.history  = []
    st.session_state.cmd_count = 0
    st.session_state.err_count = 0
    st.rerun()
```

# ── Main ───────────────────────────────────────────────────────────────────────

ns  = st.session_state.selected_ns
pod = st.session_state.selected_pod
ctr = st.session_state.selected_container
sid = st.session_state.session_id

# Badges

if sid and pod:
st.markdown(
f’<span class="badge badge-ns">ns: {ns}</span>’
f’<span class="badge badge-pod">pod: {pod}</span>’
f’<span class="badge badge-ctr">ctr: {ctr}</span>’,
unsafe_allow_html=True,
)
st.markdown(””)
else:
st.markdown(
‘<span style="color:#484f58;font-family:JetBrains Mono,monospace;font-size:13px;">’
‘Chargez les namespaces → sélectionnez un pod → ▶ Ouvrir le terminal</span>’,
unsafe_allow_html=True,
)
st.markdown(””)

# Fenêtre terminal

tab_label = f”{pod} — {ctr}” if sid else “terminal”
body_html  = render_terminal()

st.markdown(f”””

<div class="terminal-window">
  <div class="terminal-titlebar">
    <span class="dot dot-r"></span>
    <span class="dot dot-y"></span>
    <span class="dot dot-g"></span>
    <span class="terminal-title">{tab_label}</span>
  </div>
  <div class="terminal-body">
    {body_html if body_html else '<span class="line-system">Prêt.</span>'}
  </div>
</div>
""", unsafe_allow_html=True)

# ── Zone de saisie ─────────────────────────────────────────────────────────────

if sid:
prompt = prompt_str()
col_p, col_i = st.columns([3, 9])

```
with col_p:
    st.markdown(
        f'<div style="padding-top:9px;font-family:JetBrains Mono,monospace;'
        f'font-size:13px;color:#3fb950;white-space:nowrap;">{prompt}</div>',
        unsafe_allow_html=True,
    )
with col_i:
    # st.text_input déclenche un rerun à chaque Entrée :
    # on compare avec last_cmd pour savoir si c'est une nouvelle soumission
    cmd = st.text_input(
        "cmd", key="cmd_field",
        placeholder="ls, cd /app, cat fichier…",
        label_visibility="collapsed",
    )

# Détection Entrée : la valeur a changé ET elle est non vide
if cmd and cmd != st.session_state.last_cmd:
    st.session_state.last_cmd = cmd
    run_command(cmd)
    st.rerun()
```

else:
st.markdown(
‘<div style="background:#010409;border:1px solid #21262d;border-radius:6px;'
'padding:10px 16px;font-family:JetBrains Mono,monospace;font-size:13px;color:#484f58;">’
‘$ — ouvre une session pour commencer</div>’,
unsafe_allow_html=True,
)

# ── Status bar ─────────────────────────────────────────────────────────────────

st.markdown(””)
c = st.session_state.cmd_count
e = st.session_state.err_count
cwd_display = st.session_state.cwd if sid else “—”

k8s_st  = ‘<span class="status-ok">● k8s</span>’
sess_st = f’<span class="status-ok">● session active</span>’ if sid   
else ‘<span style="color:#484f58">○ idle</span>’
cwd_st  = f’<span style="color:#d29922">📁 {cwd_display}</span>’
err_col = “status-err” if e > 0 else “”
err_st  = f’<span class="{err_col}">{e} err</span>’

st.markdown(f”””

<div class="status-bar">
  {k8s_st} &nbsp;|&nbsp; {sess_st} &nbsp;|&nbsp; {cwd_st} &nbsp;|&nbsp;
  <span>{c} cmd(s)</span> &nbsp;|&nbsp; {err_st}
</div>
""", unsafe_allow_html=True)