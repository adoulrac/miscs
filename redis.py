from dash import Dash, dcc, html, Input, Output, State
from dash.dependencies import ClientsideFunction
from flask import Flask, redirect, url_for
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user
import datetime

# Initialize Flask
server = Flask(__name__)
server.secret_key = 'your_secret_key'

# Initialize Dash
app = Dash(__name__, server=server)

# Setup Flask-Login
login_manager = LoginManager()
login_manager.init_app(server)

# User class
class User(UserMixin):
    def __init__(self, id):
        self.id = id

# User loader
@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

# Simulated login route
@server.route('/login')
def login():
    user = User(id='example_user')
    login_user(user)
    return 'Logged in!'

# Simulated logout route
@server.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('login'))

app.layout = html.Div([
    html.Div(id='user-status'),
    dcc.Interval(id='interval-component', interval=5*60*1000, n_intervals=0),  # 5 minutes interval
    dcc.Store(id='last-active-timestamp'),
    html.Div(id='timeout-modal', style={'display': 'none'}, children=[
        html.Div('You have been inactive. Please confirm you are still here by clicking below.'),
        html.Button('I am here', id='confirm-button')
    ]),
    dcc.Location(id='url', refresh=True)
])

# Update user status based on activity
@app.callback(
    Output('user-status', 'children'),
    [Input('interval-component', 'n_intervals')],
    [State('last-active-timestamp', 'data')]
)
def check_inactivity(n_intervals, last_active_timestamp):
    if not current_user.is_authenticated:
        return 'Not logged in'
    
    now = datetime.datetime.now()
    if last_active_timestamp:
        last_active = datetime.datetime.fromisoformat(last_active_timestamp)
        inactivity_period = (now - last_active).total_seconds()
        if inactivity_period > 8 * 60:  # 8 minutes
            return 'Inactive: Please confirm you are still here'
    return 'User is active'

# Show timeout modal if user is inactive for too long
@app.callback(
    Output('timeout-modal', 'style'),
    [Input('user-status', 'children')]
)
def show_timeout_modal(status):
    if 'Inactive' in status:
        return {'display': 'block'}
    return {'display': 'none'}

# Update last active timestamp when user confirms they are here
@app.callback(
    Output('last-active-timestamp', 'data'),
    [Input('confirm-button', 'n_clicks')],
    [State('last-active-timestamp', 'data')]
)
def update_last_active(n_clicks, last_active_timestamp):
    return datetime.datetime.now().isoformat()

# Redirect if user does not confirm presence
@app.callback(
    Output('url', 'pathname'),
    [Input('interval-component', 'n_intervals')],
    [State('last-active-timestamp', 'data')]
)
def logout_if_inactive(n_intervals, last_active_timestamp):
    if not current_user.is_authenticated:
        return '/login'
    
    now = datetime.datetime.now()
    if last_active_timestamp:
        last_active = datetime.datetime.fromisoformat(last_active_timestamp)
        inactivity_period = (now - last_active).total_seconds()
        if inactivity_period > 10 * 60:  # 10 minutes
            logout_user()
            return '/logout'
    return '/'

if __name__ == '__main__':
    app.run_server(debug=True)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
from dash import Dash, html, dcc
import dash
from flask import Flask, redirect, url_for, session
from authlib.integrations.flask_client import OAuth
from datetime import timedelta

# Flask server setup
server = Flask(__name__)
app = Dash(__name__, server=server, suppress_callback_exceptions=True)
app.server.secret_key = 'YOUR_SECRET_KEY'

# Authlib client setup
oauth = OAuth(app.server)
oauth.register(
    name='auth_provider',
    client_id='YOUR_CLIENT_ID',
    client_secret='YOUR_CLIENT_SECRET',
    access_token_url='https://YOUR_AUTH_PROVIDER.com/oauth/token',
    authorize_url='https://YOUR_AUTH_PROVIDER.com/oauth/authorize',
    redirect_uri='http://localhost:8050/auth',
    client_kwargs={'scope': 'openid profile email'}
)

@app.server.route('/login')
def login():
    redirect_uri = url_for('authorize', _external=True)
    return oauth.auth_provider.authorize_redirect(redirect_uri)

@app.server.route('/auth')
def authorize():
    token = oauth.auth_provider.authorize_access_token()
    user = oauth.auth_provider.parse_id_token(token)
    session['user'] = user
    return redirect('/')

@app.server.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/')

# Session timeout settings
@app.server.before_request
def before_request():
    session.permanent = True
    app.server.permanent_session_lifetime = timedelta(minutes=30)
    session.modified = True

# Dash layout and callbacks
app.layout = html.Div([
    html.H1('SSO Dash App'),
    html.Div(id='user-info'),
    dcc.Location(id='url', refresh=True)
])

@app.callback(dash.dependencies.Output('user-info', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_user_info(pathname):
    user = session.get('user', None)
    if user:
        return f'Logged in as: {user["name"]}'
    return 'You are not logged in.'

if __name__ == '__main__':
    app.run_server(debug=True)
    
    
    
    
    
    
    
    
    
    
    
    
import websocket
import json

def authenticate_and_check_entitlement(username, app_id, instruments_services):
    ws_url = "wss://your-trep-websocket-endpoint"

    def send_request(ws, request):
        ws.send(json.dumps(request))
        return json.loads(ws.recv())

    try:
        ws = websocket.WebSocket()
        ws.connect(ws_url)

        # Step 1: Authenticate the user
        auth_request = {
            "ID": 1,
            "Domain": "Login",
            "Key": {
                "Name": username,
                "Elements": {
                    "ApplicationId": app_id
                }
            }
        }
        auth_response = send_request(ws, auth_request)

        if auth_response["State"]["Stream"] != "Open" or auth_response["State"]["Data"] != "Ok":
            print("Authentication failed")
            return False

        print("Authenticated successfully")

        # Step 2: Check entitlements for each instrument and service
        entitlements_check = {}
        for instrument, service in instruments_services:
            dacs_request = {
                "ID": 2,
                "Domain": "Source",
                "Key": {
                    "Name": instrument,
                    "Service": service
                }
            }
            dacs_response = send_request(ws, dacs_request)

            if dacs_response["State"]["Stream"] == "Open" and dacs_response["State"]["Data"] == "Ok":
                entitlements_check[(instrument, service)] = True
            else:
                entitlements_check[(instrument, service)] = False

        ws.close()

        return entitlements_check

    except Exception as e:
        print(f"An error occurred: {e}")
        return False

# Example usage
username = "your_username"
app_id = "your_app_id"
instruments_services = [("AAPL", "Bloomberg_feed"), ("GOOGL", "IDN_selectfeed"), ("MSFT", "Bloomberg_feed")]

entitlements = authenticate_and_check_entitlement(username, app_id, instruments_services)
print(entitlements)