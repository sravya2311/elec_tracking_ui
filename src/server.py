from flask import Flask, jsonify
from flask_cors import CORS
import subprocess
import os
import signal
import sys

app = Flask(__name__)
CORS(app)

streamlit_process = None

@app.route('/start-streamlit', methods=['POST'])
def start_streamlit():
    global streamlit_process
    try:
        if streamlit_process is None:
            # Start the Streamlit app
            streamlit_process = subprocess.Popen(
                [sys.executable, 'solar_2_5_10.py'],
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
        return jsonify({'message': 'Streamlit server started successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stop-streamlit', methods=['POST'])
def stop_streamlit():
    global streamlit_process
    try:
        if streamlit_process is not None:
            os.kill(streamlit_process.pid, signal.SIGTERM)
            streamlit_process = None
        return jsonify({'message': 'Streamlit server stopped successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000) 