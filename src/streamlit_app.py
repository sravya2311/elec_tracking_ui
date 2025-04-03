import streamlit as st
import time
from web3 import Web3
from contractInteraction import contractABI, contractAddress
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# Initialize Web3
w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:8545'))  # Connect to your local node

# Create contract instance
contract = w3.eth.contract(address=contractAddress, abi=contractABI)

# Set page config
st.set_page_config(
    page_title="Energy Trading Dashboard",
    page_icon="⚡",
    layout="wide"
)

# Title
st.title("⚡ Energy Trading Dashboard")

# Function to get all users' data
def get_all_users_data():
    try:
        # Get the EnergyDataUpdated events
        events = contract.events.EnergyDataUpdated.get_logs(fromBlock=0)
        
        # Create a dictionary to store the latest data for each user
        user_data = {}
        
        for event in events:
            user = event['args']['user']
            consumed = event['args']['consumed']
            generated = event['args']['generated']
            
            # Convert from Wei to kWh (assuming the values are stored in Wh)
            consumed_kwh = float(consumed) / 1000
            generated_kwh = float(generated) / 1000
            
            user_data[user] = {
                'consumed': consumed_kwh,
                'generated': generated_kwh,
                'timestamp': datetime.fromtimestamp(event['blockTimestamp'])
            }
        
        return user_data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return {}

# Function to create energy balance chart
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
        marker_color='red'
    ))
    
    fig.add_trace(go.Bar(
        name='Energy Generated',
        x=users,
        y=generated,
        marker_color='green'
    ))
    
    fig.update_layout(
        title='Energy Balance by User',
        xaxis_title='User Address',
        yaxis_title='Energy (kWh)',
        barmode='group'
    )
    
    return fig

# Main app loop
def main():
    # Sidebar
    st.sidebar.title("Dashboard Controls")
    refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 1, 60, 5)
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Real-time Energy Data")
        user_data = get_all_users_data()
        
        if user_data:
            # Create DataFrame for the table
            df_data = []
            for user, data in user_data.items():
                df_data.append({
                    'User': user[:10] + '...' + user[-8:],
                    'Consumed (kWh)': round(data['consumed'], 2),
                    'Generated (kWh)': round(data['generated'], 2),
                    'Last Updated': data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                })
            
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No data available yet. Connect your wallet and update energy data in the React app.")
    
    with col2:
        st.subheader("Energy Balance Chart")
        if user_data:
            fig = create_energy_chart(user_data)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Chart will appear when data is available")

    # Auto-refresh
    time.sleep(refresh_rate)
    st.experimental_rerun()

if __name__ == "__main__":
    main() 