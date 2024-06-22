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