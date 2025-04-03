import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from web3 import Web3
from datetime import datetime
import time
from contractInteraction import contractABI, contractAddress
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import joblib

# Set page configuration
st.set_page_config(
    page_title="Solar Energy Dashboard",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
        background-color: #1a1a1a;
    }
    .stMetric {
        background-color: #2d3436;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        color: #ffffff;
    }
    .stMetric:hover {
        background-color: #34495e;
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
    .stMetric label {
        color: #ffffff !important;
    }
    .stPlotlyChart {
        background-color: #2d3436;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        margin-bottom: 0px;
    }
    h1 {
        color: #ffffff;
        text-align: center;
        padding: 20px;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    h3 {
        color: #ffffff;
        padding: 15px 0;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 0;
    }
    h4 {
        color: #ffffff;
        font-size: 1.2rem;
        font-weight: 500;
        margin: 0;
    }
    .stAlert {
        border-radius: 10px;
    }
    .dashboard-container {
        background: linear-gradient(135deg, #2d3436 0%, #2c3e50 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    .metric-container {
        background: linear-gradient(135deg, #2d3436 0%, #2c3e50 100%);
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 0px;
    }
    .chart-container {
        background: linear-gradient(135deg, #2d3436 0%, #2c3e50 100%);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        margin-bottom: 0px;
    }
    .header-container {
        background: linear-gradient(120deg, #000428, #004e92);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    .header-text {
        color: #ffffff;
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .subheader-text {
        color: #ffffff;
        text-align: center;
        font-size: 1.2rem;
        opacity: 0.9;
        margin-top: 10px;
    }
    .tab-container {
        background: linear-gradient(135deg, #2d3436 0%, #2c3e50 100%);
        padding: 15px;
        border-radius: 10px;
        margin-top: 10px;
    }
    .stTabs {
        background: linear-gradient(135deg, #2d3436 0%, #2c3e50 100%);
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .stTab {
        background-color: #34495e;
        border-radius: 5px;
        padding: 5px 10px;
        margin-right: 5px;
        color: #ffffff !important;
    }
    .stTab:hover {
        background-color: #2c3e50;
    }
    div[data-testid="stExpander"] {
        background: linear-gradient(135deg, #2d3436 0%, #2c3e50 100%);
        border-radius: 10px;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        color: #ffffff;
    }
    .streamlit-expanderHeader {
        color: #ffffff !important;
        font-weight: 600;
    }
    .stMarkdown {
        color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize Web3
w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:8545'))  # Connect to your local node
contract = w3.eth.contract(address=contractAddress, abi=contractABI)

# Load the static dataset
def load_data():
    data = pd.read_csv('Plant_1_Generation_Data.csv')
    data['DATE_TIME'] = pd.to_datetime(data['DATE_TIME'], format='%d-%m-%Y %H:%M')
    data = data.sort_values('DATE_TIME')
    return data

def format_energy(value):
    if abs(value) < 1000:
        return f"{value:.2f} kWh"
    else:
        return f"{value/1000:.2f} MWh"

def create_gauge_chart(value, max_value, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'color': '#ffffff'}},
        gauge={
            'axis': {'range': [0, max_value], 'tickfont': {'color': '#ffffff'}},
            'bar': {'color': "#3498db"},
            'steps': [
                {'range': [0, max_value/3], 'color': "#2d3436"},
                {'range': [max_value/3, max_value*2/3], 'color': "#34495e"},
                {'range': [max_value*2/3, max_value], 'color': "#2c3e50"}
            ],
            'threshold': {
                'line': {'color': "#ffffff", 'width': 2},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    fig.update_layout(
        height=200,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#ffffff'}
    )
    return fig

def get_contract_data():
    try:
        # Get the EnergyDataUpdated events
        events = contract.events.EnergyDataUpdated.get_logs(fromBlock=0)
        
        # Create a dictionary to store all transactions for each user
        user_data = {}
        transaction_history = []
        
        for event in events:
            user = event['args']['user']
            consumed = event['args']['consumed']
            generated = event['args']['generated']
            
            # Convert from Wei to kWh (assuming the values are stored in Wh)
            consumed_kwh = float(consumed) / 1000
            generated_kwh = float(generated) / 1000
            
            # Get block timestamp if available, otherwise use current time
            timestamp = datetime.fromtimestamp(event.get('blockTimestamp', time.time()))
            
            # Initialize user data if not exists
            if user not in user_data:
                user_data[user] = {
                    'consumed': 0,
                    'generated': 0,
                    'timestamp': timestamp
                }
            
            # Accumulate the values
            user_data[user]['consumed'] += consumed_kwh
            user_data[user]['generated'] += generated_kwh
            user_data[user]['timestamp'] = timestamp  # Update timestamp to latest
            
            # Store transaction history
            transaction_history.append({
                'User': user,
                'Consumed (kWh)': round(consumed_kwh, 2),
                'Generated (kWh)': round(generated_kwh, 2),
                'Timestamp': timestamp,
                'Transaction Hash': event['transactionHash'].hex()
            })
        
        # Sort transaction history by timestamp (newest first)
        transaction_history.sort(key=lambda x: x['Timestamp'], reverse=True)
        
        return user_data, transaction_history
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return {}, []

def create_energy_chart(user_data):
    if not user_data:
        return None
    
    users = list(user_data.keys())
    consumed = [user_data[user]['consumed'] for user in users]
    generated = [user_data[user]['generated'] for user in users]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Energy Consumed',
        x=users,
        y=consumed,
        marker_color='#e74c3c',
        opacity=0.8,
        hovertemplate='<b>%{x}</b><br>Consumed: %{y:.2f} kWh<br><extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name='Energy Generated',
        x=users,
        y=generated,
        marker_color='#2ecc71',
        opacity=0.8,
        hovertemplate='<b>%{x}</b><br>Generated: %{y:.2f} kWh<br><extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': 'Energy Balance by User',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24, 'color': '#ffffff'}
        },
        xaxis_title='User Address',
        yaxis_title='Energy (kWh)',
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#ffffff', 'size': 12},
        legend={
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': 1.02,
            'xanchor': 'right',
            'x': 1,
            'bgcolor': 'rgba(0,0,0,0)',
            'font': {'size': 14}
        },
        margin={'t': 80, 'b': 60, 'l': 60, 'r': 40},
        height=400
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(255,255,255,0.1)',
        tickangle=45
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(255,255,255,0.1)'
    )
    
    return fig

def create_timeline_chart(user_data):
    if not user_data:
        return None
    
    # Create a DataFrame with all events
    df_data = []
    for user, data in user_data.items():
        df_data.append({
            'User': user[:10] + '...' + user[-8:],
            'Timestamp': data['timestamp'],
            'Consumed': data['consumed'],
            'Generated': data['generated']
        })
    
    df = pd.DataFrame(df_data)
    
    fig = go.Figure()
    
    # Add consumed energy trace
    fig.add_trace(go.Scatter(
        x=df['Timestamp'],
        y=df['Consumed'],
        name='Consumed',
        mode='markers+lines',
        line=dict(color='#e74c3c', width=2),
        marker=dict(size=8, color='#e74c3c'),
        hovertemplate='<b>%{x}</b><br>Consumed: %{y:.2f} kWh<br><extra></extra>'
    ))
    
    # Add generated energy trace
    fig.add_trace(go.Scatter(
        x=df['Timestamp'],
        y=df['Generated'],
        name='Generated',
        mode='markers+lines',
        line=dict(color='#2ecc71', width=2),
        marker=dict(size=8, color='#2ecc71'),
        hovertemplate='<b>%{x}</b><br>Generated: %{y:.2f} kWh<br><extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': 'Energy Timeline',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24, 'color': '#ffffff'}
        },
        xaxis_title='Time',
        yaxis_title='Energy (kWh)',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#ffffff', 'size': 12},
        legend={
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': 1.02,
            'xanchor': 'right',
            'x': 1,
            'bgcolor': 'rgba(0,0,0,0)',
            'font': {'size': 14}
        },
        margin={'t': 80, 'b': 60, 'l': 60, 'r': 40},
        height=400
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(255,255,255,0.1)',
        tickangle=45
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(255,255,255,0.1)'
    )
    
    return fig

def show_static_analysis(data):
    st.markdown("<h3 style='text-align: center;'>Historical Solar Plant Analysis</h3>", unsafe_allow_html=True)
    
    # Sidebar controls for static analysis
    with st.sidebar.expander("📊 Analysis Controls", expanded=True):
        selected_date = st.date_input(
            "Select Date",
            min_value=data['DATE_TIME'].dt.date.min(),
            max_value=data['DATE_TIME'].dt.date.max(),
            value=data['DATE_TIME'].dt.date.max()
        )
        
        show_forecast = st.checkbox("Show Forecast", value=True)
    
    # Filter data for selected date
    daily_data = data[data['DATE_TIME'].dt.date == selected_date]
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Daily Generation", format_energy(daily_data['DC_POWER'].sum()))
    with col2:
        st.metric("Daily Yield", format_energy(daily_data['DAILY_YIELD'].iloc[-1]))
    with col3:
        st.metric("Total Yield", format_energy(daily_data['TOTAL_YIELD'].iloc[-1]))
    
    # Create hourly generation chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=daily_data['DATE_TIME'],
        y=daily_data['DC_POWER'],
        name='DC Power',
        line=dict(color='#2ecc71', width=3),
        fill='tozeroy',
        fillcolor='rgba(46, 204, 113, 0.2)',
        hovertemplate='<b>Time:</b> %{x}<br>' +
                     '<b>Power:</b> %{y:.2f} kW<br>' +
                     '<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': 'Hourly Generation Pattern',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24, 'color': '#ffffff'}
        },
        xaxis_title='Time',
        yaxis_title='Power (kW)',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#ffffff', 'size': 12},
        legend={
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': 1.02,
            'xanchor': 'right',
            'x': 1,
            'bgcolor': 'rgba(0,0,0,0)',
            'font': {'size': 14}
        },
        margin={'t': 80, 'b': 60, 'l': 60, 'r': 40},
        height=400
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(255,255,255,0.1)',
        tickangle=45
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(255,255,255,0.1)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    if show_forecast:
        # Create forecast visualization
        st.markdown("<h3 style='text-align: center;'>Generation Forecast</h3>", unsafe_allow_html=True)
        
        # Generate forecast data (simplified example)
        future_hours = np.arange(24)
        base_pattern = np.sin(2 * np.pi * future_hours / 24) + 1
        forecast = base_pattern * daily_data['DC_POWER'].mean()
        
        fig = go.Figure()
        
        # Add actual data for comparison
        fig.add_trace(go.Scatter(
            x=daily_data['DATE_TIME'],
            y=daily_data['DC_POWER'],
            name='Actual Power',
            line=dict(color='#2ecc71', width=2),
            hovertemplate='<b>Time:</b> %{x}<br>' +
                         '<b>Power:</b> %{y:.2f} kW<br>' +
                         '<extra></extra>'
        ))
        
        # Add forecast data
        future_times = pd.date_range(
            start=daily_data['DATE_TIME'].iloc[-1],
            periods=25,
            freq='H'
        )[1:]  # Exclude the last actual data point
        
        fig.add_trace(go.Scatter(
            x=future_times,
            y=forecast,
            name='Forecast',
            line=dict(color='#3498db', width=3, dash='dash'),
            fill='tozeroy',
            fillcolor='rgba(52, 152, 219, 0.2)',
            hovertemplate='<b>Time:</b> %{x}<br>' +
                         '<b>Forecast:</b> %{y:.2f} kW<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': '24-Hour Generation Forecast',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 24, 'color': '#ffffff'}
            },
            xaxis_title='Time',
            yaxis_title='Power (kW)',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#ffffff', 'size': 12},
            legend={
                'orientation': 'h',
                'yanchor': 'bottom',
                'y': 1.02,
                'xanchor': 'right',
                'x': 1,
                'bgcolor': 'rgba(0,0,0,0)',
                'font': {'size': 14}
            },
            margin={'t': 80, 'b': 60, 'l': 60, 'r': 40},
            height=400
        )
        
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(255,255,255,0.1)',
            tickangle=45
        )
        
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(255,255,255,0.1)'
        )
        
        st.plotly_chart(fig, use_container_width=True)

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, 1)  # *2 for bidirectional
        
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # *2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Get the last time step output
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

def prepare_data_for_lstm(data, sequence_length=4):  # Reduced sequence length
    """Prepare data for LSTM model"""
    # Extract features
    data['hour'] = data['Timestamp'].dt.hour
    data['day_of_week'] = data['Timestamp'].dt.dayofweek
    data['month'] = data['Timestamp'].dt.month
    
    # Scale the features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Consumed (kWh)', 'hour', 'day_of_week', 'month']])
    
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:(i + sequence_length)])
        y.append(scaled_data[i + sequence_length, 0])  # Predict only consumption
    
    return np.array(X), np.array(y), scaler

def train_and_predict_energy_consumption(transaction_history):
    """Train Bi-LSTM model and make predictions"""
    if not transaction_history:
        return None, None
    
    # Convert transaction history to DataFrame
    df = pd.DataFrame(transaction_history)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Prepare data with reduced sequence length
    sequence_length = min(4, len(df) - 1)  # Use smaller sequence length based on available data
    X, y, scaler = prepare_data_for_lstm(df, sequence_length)
    
    if len(X) < 2:  # Need at least 2 samples for training
        return None, None
    
    # Split data
    train_size = max(1, int(len(X) * 0.8))  # Ensure at least 1 training sample
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)
    
    # Create and train model
    input_size = X.shape[2]  # Number of features
    model = BiLSTM(input_size=input_size, hidden_size=32, num_layers=1)  # Simplified model
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Increased learning rate
    
    # Training loop with fewer epochs
    num_epochs = 20
    batch_size = min(32, len(X_train))  # Adjust batch size based on data
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Calculate validation loss
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            val_loss = criterion(val_outputs.squeeze(), y_test)
        
        train_losses.append(total_loss / (len(X_train) // batch_size))
        val_losses.append(val_loss.item())
    
    # Make predictions for next few hours
    model.eval()
    with torch.no_grad():
        last_sequence = torch.FloatTensor(X[-1]).unsqueeze(0)
        future_predictions = []
        
        # Predict next few hours based on available data
        prediction_hours = min(4, len(df))  # Predict up to 4 hours or less if less data
        
        for _ in range(prediction_hours):
            # Predict next hour
            next_pred = model(last_sequence)
            future_predictions.append(next_pred.item())
            
            # Update sequence for next prediction
            last_sequence = torch.roll(last_sequence, -1, dims=1)
            last_sequence[0, -1, 0] = next_pred.item()
    
    # Inverse transform predictions
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = scaler.inverse_transform(
        np.hstack([future_predictions, np.zeros((len(future_predictions), 3))])
    )[:, 0]
    
    return future_predictions, {'train_loss': train_losses[-1], 'val_loss': val_losses[-1]}

def show_ml_predictions(transaction_history):
    """Display ML predictions in Streamlit"""
    st.markdown("<h3 style='text-align: center;'>ML Energy Consumption Predictions</h3>", unsafe_allow_html=True)
    
    if not transaction_history:
        st.info("No transaction history available for predictions.")
        return
    
    # Train model and get predictions
    predictions, losses = train_and_predict_energy_consumption(transaction_history)
    
    if predictions is None:
        st.info("Not enough data for predictions. Need at least 2 data points.")
        return
    
    # Create future timestamps
    last_timestamp = pd.to_datetime(transaction_history[-1]['Timestamp'])
    future_timestamps = pd.date_range(start=last_timestamp, periods=len(predictions) + 1, freq='H')[1:]
    
    # Create prediction chart
    fig = go.Figure()
    
    # Add historical data
    historical_data = pd.DataFrame(transaction_history)
    historical_data['Timestamp'] = pd.to_datetime(historical_data['Timestamp'])
    
    fig.add_trace(go.Scatter(
        x=historical_data['Timestamp'],
        y=historical_data['Consumed (kWh)'],
        name='Historical Consumption',
        line=dict(color='#2ecc71', width=2),
        hovertemplate='<b>Time:</b> %{x}<br>' +
                     '<b>Consumption:</b> %{y:.2f} kWh<br>' +
                     '<extra></extra>'
    ))
    
    # Add predictions
    fig.add_trace(go.Scatter(
        x=future_timestamps,
        y=predictions,
        name='Predictions',
        line=dict(color='#3498db', width=3, dash='dash'),
        fill='tozeroy',
        fillcolor='rgba(52, 152, 219, 0.2)',
        hovertemplate='<b>Time:</b> %{x}<br>' +
                     '<b>Predicted Consumption:</b> %{y:.2f} kWh<br>' +
                     '<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': f'{len(predictions)}-Hour Energy Consumption Forecast',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24, 'color': '#ffffff'}
        },
        xaxis_title='Time',
        yaxis_title='Energy Consumption (kWh)',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#ffffff', 'size': 12},
        legend={
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': 1.02,
            'xanchor': 'right',
            'x': 1,
            'bgcolor': 'rgba(0,0,0,0)',
            'font': {'size': 14}
        },
        margin={'t': 80, 'b': 60, 'l': 60, 'r': 40},
        height=400
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(255,255,255,0.1)',
        tickangle=45
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(255,255,255,0.1)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display model metrics
    if losses:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Training Loss", f"{losses['train_loss']:.4f}")
        with col2:
            st.metric("Validation Loss", f"{losses['val_loss']:.4f}")

def show_realtime_monitoring():
    st.markdown("<h3 style='text-align: center;'>Real-time Blockchain Data</h3>", unsafe_allow_html=True)
    
    # Refresh rate control
    refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 1, 60, 5)
    
    # Get real-time data from the contract
    user_data, transaction_history = get_contract_data()
    
    if user_data:
        # Calculate total energy metrics
        total_consumed = sum(data['consumed'] for data in user_data.values())
        total_generated = sum(data['generated'] for data in user_data.values())
        net_energy = total_generated - total_consumed
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Energy Generated", format_energy(total_generated))
        with col2:
            st.metric("Total Energy Consumed", format_energy(total_consumed))
        with col3:
            st.metric("Net Energy Balance", format_energy(net_energy))
        
        # Display charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_energy_chart(user_data), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_timeline_chart(user_data), use_container_width=True)
        
        # Add ML predictions section
        show_ml_predictions(transaction_history)
        
        # Display transaction history
        st.subheader("Transaction History")
        
        # Add filters for transaction history
        col1, col2 = st.columns(2)
        with col1:
            selected_user = st.selectbox(
                "Filter by User",
                ["All"] + list(set(tx['User'] for tx in transaction_history))
            )
        with col2:
            date_range = st.date_input(
                "Date Range",
                value=(
                    min(tx['Timestamp'].date() for tx in transaction_history),
                    max(tx['Timestamp'].date() for tx in transaction_history)
                )
            )
        
        # Filter transactions
        filtered_transactions = transaction_history
        if selected_user != "All":
            filtered_transactions = [tx for tx in filtered_transactions if tx['User'] == selected_user]
        if len(date_range) == 2:
            filtered_transactions = [
                tx for tx in filtered_transactions 
                if date_range[0] <= tx['Timestamp'].date() <= date_range[1]
            ]
        
        # Display filtered transactions
        if filtered_transactions:
            df = pd.DataFrame(filtered_transactions)
            df['User'] = df['User'].apply(lambda x: x[:10] + '...' + x[-8:])
            df['Timestamp'] = df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            df['Transaction Hash'] = df['Transaction Hash'].apply(
                lambda x: f"[{x[:10]}...{x[-8:]}](https://sepolia.etherscan.io/tx/{x})"
            )
            
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No transactions found for the selected filters.")
    else:
        st.info("No data available yet. Connect your wallet and update energy data in the React app.")

# Set up the Streamlit app
def main():
    # Header with gradient background
    st.markdown("""
        <div class="header-container">
            <div class="header-text">⚡ Solar Energy Dashboard</div>
            <div class="subheader-text">
                Analyze historical data and monitor real-time energy trading
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar configuration
    st.sidebar.image("https://img.icons8.com/fluency/96/000000/solar-panel.png", width=100)
    st.sidebar.title("Control Panel")
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["📊 Historical Analysis", "⚡ Real-time Monitoring"])
    
    # Load static data
    data = load_data()
    
    with tab1:
        show_static_analysis(data)
    
    with tab2:
        show_realtime_monitoring()
    
    # Auto-refresh for real-time monitoring
    time.sleep(5)  # Default refresh rate
    st.experimental_rerun()

if __name__ == "__main__":
    main()
