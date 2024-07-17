
def deep_merge(dict1, dict2):
    """Recursively merge two dictionaries."""
    for key in dict2:
        if key in dict1:
            if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                deep_merge(dict1[key], dict2[key])
            else:
                # If there is a conflict, you can decide which one to keep.
                # Here we choose the value from dict2, but you can customize as needed.
                dict1[key] = dict2[key]
        else:
            dict1[key] = dict2[key]
    return dict1

def combine_dicts(dicts):
    """Combine a list of dict of dict of dict."""
    combined_dict = {}
    for d in dicts:
        deep_merge(combined_dict, d)
    return combined_dict

# Example usage
dict1 = {
    'a': {'b': {'c': 1}},
    'd': {'e': {'f': 2}}
}

dict2 = {
    'a': {'b': {'d': 3}},
    'd': {'g': {'h': 4}}
}

dict3 = {
    'a': {'i': {'j': 5}},
    'k': {'l': {'m': 6}}
}

combined = combine_dicts([dict1, dict2, dict3])
print(combined)



If your dictionaries are ordered, you can leverage this property to improve performance, particularly when splitting the data. Since ordered dictionaries maintain the insertion order, you can stop iterating as soon as you encounter a timestamp that doesn't meet the criteria, avoiding unnecessary iterations.

Here’s how you can modify the functions to take advantage of the ordered nature of your dictionaries:

### Split Function

```python
from collections import OrderedDict
import datetime

def split_data_dict(data_dict):
    one_hour_ago_ts = int((datetime.datetime.now() - datetime.timedelta(hours=1)).timestamp() * 1000)
    last_hour_data, older_data = OrderedDict(), OrderedDict()

    for level1_key, level2_dict in data_dict.items():
        last_hour_data[level1_key], older_data[level1_key] = OrderedDict(), OrderedDict()
        for level2_key, level3_dict in level2_dict.items():
            last_hour_data[level1_key][level2_key], older_data[level1_key][level2_key] = OrderedDict(), OrderedDict()
            for timestamp, value in level3_dict.items():
                if int(timestamp) >= one_hour_ago_ts:
                    last_hour_data[level1_key][level2_key][timestamp] = value
                else:
                    older_data[level1_key][level2_key][timestamp] = value

            # Clean up empty dictionaries
            if not last_hour_data[level1_key][level2_key]:
                del last_hour_data[level1_key][level2_key]
            if not older_data[level1_key][level2_key]:
                del older_data[level1_key][level2_key]

        # Clean up empty dictionaries
        if not last_hour_data[level1_key]:
            del last_hour_data[level1_key]
        if not older_data[level1_key]:
            del older_data[level1_key]

    return last_hour_data, older_data
```

### Combine Function

```python
def combine_data_dict(last_hour_data, older_data):
    combined_data = older_data.copy()

    for level1_key, level2_dict in last_hour_data.items():
        if level1_key not in combined_data:
            combined_data[level1_key] = OrderedDict()
        for level2_key, level3_dict in level2_dict.items():
            if level2_key not in combined_data[level1_key]:
                combined_data[level1_key][level2_key] = OrderedDict()
            combined_data[level1_key][level2_key].update(level3_dict)
    
    return combined_data
```

### Usage Example

Here’s how to use these functions:

```python
import time
import threading
from collections import OrderedDict

# Example nested dictionary structure with ordered data
data_dict = OrderedDict({
    'key1': OrderedDict({
        'subkey1': OrderedDict({
            '1625010000000': 'value1',
            '1625013600000': 'value2'
        }),
        'subkey2': OrderedDict({
            '1625017200000': 'value3',
            '1625020800000': 'value4'
        })
    }),
    'key2': OrderedDict({
        'subkey3': OrderedDict({
            '1625024400000': 'value5',
            '1625028000000': 'value6'
        })
    })
})

# Split data into last hour and older
last_hour_data, older_data = split_data_dict(data_dict)

# Combine data back into one dictionary
combined_data = combine_data_dict(last_hour_data, older_data)

# Background task for continuous update
def background_task():
    while True:
        global data_dict, older_data
        last_hour_data, older_data = split_data_dict(data_dict)
        data_dict = last_hour_data
        time.sleep(3600)

thread = threading.Thread(target=background_task)
thread.start()
```

### Explanation

1. **split_data_dict**:
    - Uses `OrderedDict` to maintain order.
    - Since dictionaries are ordered, the function can efficiently split the data without unnecessary iterations.

2. **combine_data_dict**:
    - Uses `OrderedDict` to maintain order.
    - Combines the dictionaries efficiently by updating only the necessary parts.

### Performance Considerations

- **Ordered Iteration**: Leveraging the ordered nature of dictionaries allows for early termination of loops, improving performance.
- **Dictionary Operations**: Efficient dictionary operations (`update`, `copy`) help maintain performance even with nested structures.
- **Concurrency**: The background task continues to run in a separate thread, ensuring the main application remains responsive.

This optimized approach should provide better performance by taking advantage of the ordered nature of your dictionaries.



Certainly! To handle your nested dictionary structure and efficiently split and combine dictionaries, we need to traverse the nested levels and perform the required operations. Here's how you can achieve this:

### Split Function

The split function will separate data older than one hour from the data within the last hour.

```python
import datetime

def split_data_dict(data_dict):
    one_hour_ago = datetime.datetime.now() - datetime.timedelta(hours=1)
    one_hour_ago_ts = int(one_hour_ago.timestamp() * 1000)  # Assuming timestamps are in milliseconds

    last_hour_data = {}
    older_data = {}

    for level1_key, level2_dict in data_dict.items():
        last_hour_data[level1_key] = {}
        older_data[level1_key] = {}

        for level2_key, level3_dict in level2_dict.items():
            last_hour_data[level1_key][level2_key] = {}
            older_data[level1_key][level2_key] = {}

            for timestamp, value in level3_dict.items():
                if int(timestamp) >= one_hour_ago_ts:
                    last_hour_data[level1_key][level2_key][timestamp] = value
                else:
                    older_data[level1_key][level2_key][timestamp] = value

            # Clean up empty dictionaries
            if not last_hour_data[level1_key][level2_key]:
                del last_hour_data[level1_key][level2_key]
            if not older_data[level1_key][level2_key]:
                del older_data[level1_key][level2_key]

        # Clean up empty dictionaries
        if not last_hour_data[level1_key]:
            del last_hour_data[level1_key]
        if not older_data[level1_key]:
            del older_data[level1_key]

    return last_hour_data, older_data
```

### Combine Function

The combine function will merge the older data and the last hour data back together.

```python
def combine_data_dict(last_hour_data, older_data):
    combined_data = older_data.copy()  # Start with older data

    for level1_key, level2_dict in last_hour_data.items():
        if level1_key not in combined_data:
            combined_data[level1_key] = {}

        for level2_key, level3_dict in level2_dict.items():
            if level2_key not in combined_data[level1_key]:
                combined_data[level1_key][level2_key] = {}

            combined_data[level1_key][level2_key].update(level3_dict)

    return combined_data
```

### Usage Example

To use these functions, you can follow this pattern in your script:

```python
import time
import datetime

# Example nested dictionary structure
data_dict = {
    'key1': {
        'subkey1': {
            'timestamp1': 'value1',
            'timestamp2': 'value2'
        },
        'subkey2': {
            'timestamp3': 'value3',
            'timestamp4': 'value4'
        }
    },
    'key2': {
        'subkey3': {
            'timestamp5': 'value5',
            'timestamp6': 'value6'
        }
    }
}

# Split data into last hour and older
last_hour_data, older_data = split_data_dict(data_dict)

# ... use last_hour_data and older_data as needed ...

# Combine data back into one dictionary
combined_data = combine_data_dict(last_hour_data, older_data)

# For continuous update example
def background_task():
    while True:
        global data_dict, older_data
        last_hour_data, older_data = split_data_dict(data_dict)
        data_dict.clear()
        data_dict.update(last_hour_data)
        time.sleep(3600)

thread = threading.Thread(target=background_task)
thread.start()
```

### Explanation

1. **split_data_dict**:
    - This function traverses the three levels of dictionaries.
    - It separates the entries based on whether the timestamp is within the last hour or older.

2. **combine_data_dict**:
    - This function merges the older data with the data from the last hour.
    - It ensures that all keys are correctly merged without overwriting existing data.

### Performance Considerations

- **Efficiency**: Both functions use efficient dictionary operations (`update`, `copy`) which are generally performant in Python.
- **Data Traversal**: Traversing the nested dictionary structure is necessary to ensure accurate splitting and combining.
- **Concurrency**: Running the background task in a separate thread ensures that your main application remains responsive.

This approach ensures that you can efficiently manage and process your nested dictionaries, maintaining high performance even with large datasets.




import requests

class OpenDACSAPI:
    def __init__(self, base_url):
        self.base_url = base_url
        self.session = requests.Session()
        self.token = None

    def login(self, username, password):
        login_url = f"{self.base_url}/login"
        payload = {
            'username': username,
            'password': password
        }
        response = self.session.post(login_url, json=payload)
        if response.status_code == 200:
            self.token = response.json().get('token')
            self.session.headers.update({'Authorization': f'Bearer {self.token}'})
            return True
        else:
            print(f"Login failed: {response.status_code} - {response.text}")
            return False

    def check_access(self, service, instrument):
        if not self.token:
            print("User not logged in.")
            return False
        
        access_url = f"{self.base_url}/access"
        payload = {
            'service': service,
            'instrument': instrument
        }
        response = self.session.post(access_url, json=payload)
        if response.status_code == 200:
            access_data = response.json()
            return access_data.get('service_access', False) and access_data.get('instrument_access', False)
        else:
            print(f"Access check failed: {response.status_code} - {response.text}")
            return False

    def logout(self):
        if not self.token:
            print("User not logged in.")
            return False

        logout_url = f"{self.base_url}/logout"
        response = self.session.post(logout_url)
        if response.status_code == 200:
            self.token = None
            self.session.headers.pop('Authorization', None)
            return True
        else:
            print(f"Logout failed: {response.status_code} - {response.text}")
            return False

# Example usage:
if __name__ == "__main__":
    base_url = "https://api.opendacs.com"  # Replace with the actual base URL
    api = OpenDACSAPI(base_url)

    if api.login("your_username", "your_password"):
        print("Login successful")
        
        service = "example_service"
        instrument = "example_instrument"
        if api.check_access(service, instrument):
            print(f"Access granted to service '{service}' and instrument '{instrument}'")
        else:
            print(f"Access denied to service '{service}' and instrument '{instrument}'")

        if api.logout():
            print("Logout successful")
        else:
            print("Logout failed")
    else:
        print("Login failed")

def generate_html_table(data, headers=None):
    html = '''
    <html>
    <head>
    <style>
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 12px;
            border: none;
        }
        th {
            background-color: transparent;
        }
        tr:nth-child(even) {
            background-color: transparent;
        }
        tr:hover {
            background-color: transparent;
        }
    </style>
    </head>
    <body>
    <table>
    '''
    
    # Add table headers if provided
    if headers:
        html += '  <tr>\n'
        for header in headers:
            html += f'    <th>{header}</th>\n'
        html += '  </tr>\n'
    
    # Add table rows
    for row in data:
        html += '  <tr>\n'
        for item in row:
            html += f'    <td>{item}</td>\n'
        html += '  </tr>\n'
    
    html += '''
    </table>
    </body>
    </html>
    '''
    return html

# Example usage
headers = ["ID", "Name", "Age"]
html_table = generate_html_table(data, headers)

# Save to file
with open("invisible_table.html", "w") as file:
    file.write(html_table)

print("Invisible HTML table saved to invisible_table.html")



def generate_html_table(data, headers=None):
    html = '<table border="1">\n'
    
    # Add table headers if provided
    if headers:
        html += '  <tr>\n'
        for header in headers:
            html += f'    <th>{header}</th>\n'
        html += '  </tr>\n'
    
    # Add table rows
    for row in data:
        html += '  <tr>\n'
        for item in row:
            html += f'    <td>{item}</td>\n'
        html += '  </tr>\n'
    
    html += '</table>'
    return html

# Example usage
headers = ["ID", "Name", "Age"]
html_table = generate_html_table(data, headers)
print(html_table)





import quickfix as fix
import quickfix44 as fix44
import time


class Application(fix.Application):
    def onCreate(self, sessionID):
        print(f"Session created: {sessionID}")

    def onLogon(self, sessionID):
        print(f"Logon: {sessionID}")
        self.subscribe_market_data(sessionID)

    def onLogout(self, sessionID):
        print(f"Logout: {sessionID}")

    def toAdmin(self, message, sessionID):
        print(f"ToAdmin: {message}")

    def fromAdmin(self, message, sessionID):
        print(f"FromAdmin: {message}")

    def toApp(self, message, sessionID):
        print(f"ToApp: {message}")

    def fromApp(self, message, sessionID):
        print(f"FromApp: {message}")
        msg_type = fix.MsgType()
        message.getHeader().getField(msg_type)
        if msg_type.getValue() == fix.MsgType_MarketDataSnapshotFullRefresh:
            self.on_market_data(message)

    def subscribe_market_data(self, sessionID):
        md_request = fix44.MarketDataRequest(
            fix.MDReqID("1"),
            fix.SubscriptionRequestType(fix.SubscriptionRequestType_SNAPSHOT),
            fix.MarketDepth(1)
        )
        md_entry_types = fix44.MarketDataRequest.NoMDEntryTypes()
        md_entry_types.setField(fix.MDEntryType(fix.MDEntryType_BID))
        md_request.addGroup(md_entry_types)
        md_entry_types.setField(fix.MDEntryType(fix.MDEntryType_OFFER))
        md_request.addGroup(md_entry_types)
        instruments = fix44.MarketDataRequest.NoRelatedSym()
        instruments.setField(fix.Symbol("instrument_name"))
        md_request.addGroup(instruments)
        fix.Session.sendToTarget(md_request, sessionID)

    def on_market_data(self, message):
        symbol = fix.Symbol()
        message.getField(symbol)
        print(f"Market Data for {symbol.getValue()}")
        no_md_entries = fix.NoMDEntries()
        message.getField(no_md_entries)
        for i in range(int(no_md_entries.getValue())):
            group = fix44.MarketDataSnapshotFullRefresh.NoMDEntries()
            message.getGroup(i + 1, group)
            entry_type = fix.MDEntryType()
            group.getField(entry_type)
            price = fix.MDEntryPx()
            group.getField(price)
            size = fix.MDEntrySize()
            group.getField(size)
            print(f"  Type: {entry_type.getValue()}, Price: {price.getValue()}, Size: {size.getValue()}")


def main():
    settings = fix.SessionSettings("quickfix.cfg")
    application = Application()
    store_factory = fix.FileStoreFactory(settings)
    log_factory = fix.FileLogFactory(settings)
    initiator = fix.SocketInitiator(application, store_factory, settings, log_factory)
    initiator.start()
    print("FIX initiator started.")
    time.sleep(60)  # Keep the connection open for 60 seconds
    initiator.stop()
    print("FIX initiator stopped.")


if __name__ == "__main__":
    main()



import dash
import dash_bootstrap_components as dbc
from dash import html

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Navbar(
        children=[
            dbc.NavbarBrand("Navbar"),
            dbc.Nav(
                [
                    dbc.NavItem(dbc.NavLink("Home", href="#")),
                    dbc.NavItem(dbc.NavLink("Page 2", href="#")),
                    dbc.NavItem(dbc.NavLink("Dropdown", href="#")),
                ],
                navbar=True,
                className="ml-auto",
            ),
            dbc.DropdownMenu(
                children=[
                    dbc.DropdownMenuItem("Item 1", href="#"),
                    dbc.DropdownMenuItem("Item 2", href="#"),
                ],
                nav=True,
                in_navbar=True,
                label="Menu",
            )
        ],
        color="primary",
        dark=True,
        className="mb-3"
    ),
    html.Div(id='page-content')
], fluid=True)

@app.callback(
    dash.dependencies.Output('page-content', 'children'),
    [dash.dependencies.Input('logout-button', 'n_clicks')]
)
def logout(n_clicks):
    if n_clicks:
        # Here you can add the logic to handle the logout, e.g., closing a session
        return html.Div("Logged out")
    return html.Div("Welcome to the app")

if __name__ == '__main__':
    app.run_server(debug=True)


import dash
import dash_bootstrap_components as dbc
from dash import html, Input, Output

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Custom CSS to style the logoff button
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .btn-logoff {
                color: white !important;
                background-color: #007bff !important; /* Blue color */
                border: none;
            }
            .btn-logoff:hover {
                background-color: #0056b3 !important; /* Darker blue on hover */
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Layout with NavbarSimple and Logoff Button
app.layout = html.Div([
    dbc.NavbarSimple(
        brand="My Dashboard",
        brand_href="#",
        color="primary",
        dark=True,
        children=[
            # Left-aligned nav items
            dbc.NavItem(dbc.NavLink("Home", href="#")),
            dbc.NavItem(dbc.NavLink("Page 1", href="#")),
            dbc.NavItem(dbc.NavLink("Page 2", href="#")),
            # Container with d-flex and ml-auto to align logoff button to the right
            html.Div(
                dbc.Button("Logoff", id="logoff-button", className="btn-logoff"),
                className="d-flex ml-auto"
            )
        ]
    ),
    html.Div(id='logoff-output')
])

# Callback to handle logoff button click
@app.callback(
    Output('logoff-output', 'children'),
    Input('logoff-button', 'n_clicks'),
    prevent_initial_call=True
)
def logoff_user(n_clicks):
    if n_clicks is not None:
        # Perform logoff actions here, e.g., closing WebSocket connection, clearing session data, etc.
        return "Logged off successfully."
    return ""

if __name__ == '__main__':
    app.run_server(debug=True)

import dash
import dash_bootstrap_components as dbc
from dash import html, Input, Output

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Custom CSS to style the logoff button
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .btn-logoff {
                color: white !important;
                background-color: #007bff !important; /* Blue color */
                border: none;
            }
            .btn-logoff:hover {
                background-color: #0056b3 !important; /* Darker blue on hover */
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Layout with NavbarSimple and Logoff Button
app.layout = html.Div([
    dbc.NavbarSimple(
        brand="My Dashboard",
        brand_href="#",
        color="primary",
        dark=True,
        children=[
            # Left-aligned nav items
            dbc.NavItem(dbc.NavLink("Home", href="#")),
            dbc.NavItem(dbc.NavLink("Page 1", href="#")),
            dbc.NavItem(dbc.NavLink("Page 2", href="#")),
            # Right-aligned logoff button using ml-auto
            dbc.Nav(
                [
                    dbc.NavItem(
                        html.A(
                            dbc.Button("Logoff", id="logoff-button", className="btn-logoff"),
                            href="/logout-url"  # Specify the URL to redirect to
                        )
                    )
                ],
                className="ml-auto",
                navbar=True,
            )
        ]
    ),
    html.Div(id='logoff-output')
])

# Callback to handle logoff button click
@app.callback(
    Output('logoff-output', 'children'),
    Input('logoff-button', 'n_clicks'),
    prevent_initial_call=True
)
def logoff_user(n_clicks):
    if n_clicks is not None:
        # Perform logoff actions here, e.g., closing WebSocket connection, clearing session data, etc.
        return "Logged off successfully."
    return ""

if __name__ == '__main__':
    app.run_server(debug=True)

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout with NavbarSimple and Logoff Button
app.layout = html.Div([
    dbc.NavbarSimple(
        brand="My Dashboard",
        brand_href="#",
        color="primary",
        dark=True,
        children=[
            dbc.NavItem(dbc.NavLink("Home", href="#")),
            dbc.NavItem(dbc.NavLink("Page 1", href="#")),
            dbc.NavItem(dbc.NavLink("Page 2", href="#")),
            dbc.NavItem(
                dbc.Button("Logoff", id="logoff-button", color="danger", className="ml-auto"),
                className="ml-auto"
            )
        ]
    ),
    html.Div(id='logoff-output')
])

# Callback to handle logoff button click
@app.callback(
    Output('logoff-output', 'children'),
    Input('logoff-button', 'n_clicks'),
    prevent_initial_call=True
)
def logoff_user(n_clicks):
    if n_clicks is not None:
        # Perform logoff actions here, e.g., closing WebSocket connection, clearing session data, etc.
        return "Logged off successfully."
    return ""

if __name__ == '__main__':
    app.run_server(debug=True)


from dash import Dash, html, dcc, Input, Output, State
import dash

app = Dash(__name__)

# Sample Dash layout with a button to trigger the confirmation dialog
app.layout = html.Div([
    html.H1('Dash Application'),
    html.Button('Delete Item', id='delete-button'),
    dcc.ConfirmDialog(
        id='confirm-dialog',
        message='Are you sure you want to delete this item?',
    ),
    html.Div(id='output')
])

# Callback to open the confirmation dialog
@app.callback(
    Output('confirm-dialog', 'displayed'),
    Input('delete-button', 'n_clicks'),
    prevent_initial_call=True
)
def display_confirm(n_clicks):
    return True  # Open the confirmation dialog

# Callback to handle the user's response to the confirmation dialog
@app.callback(
    Output('output', 'children'),
    Input('confirm-dialog', 'submit_n_clicks'),
    State('confirm-dialog', 'submit_n_clicks_timestamp'),
    prevent_initial_call=True
)
def handle_confirm(submit_n_clicks, submit_n_clicks_timestamp):
    if submit_n_clicks is not None:
        return "Item deleted." if submit_n_clicks > 0 else "Deletion cancelled."
    return ""

if __name__ == '__main__':
    app.run_server(debug=True)


from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.wsgi import WSGIMiddleware
from dash import Dash, html, dcc, Input, Output
import uvicorn
import threading
import websocket
import json

app = FastAPI()

# Dictionary to store user approval status
user_approval_status = {}

# WebSocket event handlers
def on_message(ws, message):
    data = json.loads(message)
    user_ip = data.get('user_ip')
    is_approved = data.get('is_approved')
    user_approval_status[user_ip] = is_approved

def on_error(ws, error):
    print(error)

def on_close(ws):
    print("### closed ###")

def on_open(ws):
    def run(*args):
        while True:
            # Keep the connection open and ignore stream data
            pass
    threading.Thread(target=run).start()

# Function to start WebSocket client
def start_websocket_client():
    websocket.enableTrace(True)
    global ws
    ws = websocket.WebSocketApp("wss://dacs-websocket-api-endpoint",
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.on_open = on_open
    ws.run_forever()

# Start WebSocket client in a separate thread
threading.Thread(target=start_websocket_client).start()

# FastAPI middleware to check user login
@app.middleware("http")
async def check_user_login(request: Request, call_next):
    user_ip = request.client.host
    if not is_user_approved(user_ip):
        raise HTTPException(status_code=401, detail="Unauthorized")
    response = await call_next(request)
    return response

# Function to check user approval status
def is_user_approved(user_ip):
    return user_approval_status.get(user_ip, False)

# Initialize Dash app
dash_app = Dash(__name__)
dash_app.layout = html.Div([
    html.H1('Dash Application'),
    html.Button('Logoff', id='logoff-button'),
    html.Div(id='logoff-output')
])

# Callback to handle logoff button click
@dash_app.callback(
    Output('logoff-output', 'children'),
    Input('logoff-button', 'n_clicks')
)
def logoff_user(n_clicks):
    if n_clicks is not None:
        # Send close request to DACS WebSocket API
        ws.close()
        return "Logged off successfully."
    return ""

# Mount Dash app to FastAPI
app.mount("/", WSGIMiddleware(dash_app.server))

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8050)
    
    
    


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