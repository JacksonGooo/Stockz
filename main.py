import os
import requests
from flask import Flask, render_template, request, send_from_directory, redirect, url_for

app = Flask(__name__)
STORAGE_DIR = os.path.join(os.path.dirname(__file__), 'storage')

def get_stocks():
	# Each stock has its own folder
	return [d for d in os.listdir(STORAGE_DIR) if os.path.isdir(os.path.join(STORAGE_DIR, d))]

def get_stock_files(stock):
	stock_path = os.path.join(STORAGE_DIR, stock)
	if not os.path.exists(stock_path):
		return []
	return sorted(os.listdir(stock_path))

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/network')
def network():
	return render_template('network.html')

@app.route('/storage', methods=['GET', 'POST'])
def storage():
	stocks = get_stocks()
	selected_stock = request.form.get('stock') if request.method == 'POST' else None
	files = get_stock_files(selected_stock) if selected_stock else []
	return render_template('storage.html', stocks=stocks, selected_stock=selected_stock, files=files)

@app.route('/storage/upload', methods=['POST'])
def upload_file():
	stock = request.form.get('stock')
	if not stock:
		return redirect(url_for('storage'))
	stock_path = os.path.join(STORAGE_DIR, stock)
	os.makedirs(stock_path, exist_ok=True)
	if 'file' not in request.files:
		return redirect(url_for('storage'))
	file = request.files['file']
	if file.filename == '':
		return redirect(url_for('storage'))
	file.save(os.path.join(stock_path, file.filename))
	return redirect(url_for('storage'))

@app.route('/storage/download/<stock>/<filename>')
def download_file(stock, filename):
	stock_path = os.path.join(STORAGE_DIR, stock)
	return send_from_directory(stock_path, filename, as_attachment=True)

# API keys (replace with your actual keys)
POLYGON_API_KEY = 'YOUR_POLYGON_API_KEY'
FINNHUB_API_KEY = 'YOUR_FINNHUB_API_KEY'
IEX_API_KEY = 'YOUR_IEX_API_KEY'

def fetch_polygon_data(symbol):
	url = f'https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/2023-01-01/2023-01-02?apiKey={POLYGON_API_KEY}'
	resp = requests.get(url)
	return resp.json()

def fetch_finnhub_data(symbol):
	url = f'https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_API_KEY}'
	resp = requests.get(url)
	return resp.json()

def fetch_iex_data(symbol):
	url = f'https://cloud.iexapis.com/stable/stock/{symbol}/quote?token={IEX_API_KEY}'
	resp = requests.get(url)
	return resp.json()

@app.route('/api/fetch_data', methods=['POST'])
def fetch_data():
	symbol = request.form.get('symbol')
	source = request.form.get('source')
	if not symbol or not source:
		return {'error': 'Missing symbol or source'}, 400
	if source == 'polygon':
		data = fetch_polygon_data(symbol)
	elif source == 'finnhub':
		data = fetch_finnhub_data(symbol)
	elif source == 'iex':
		data = fetch_iex_data(symbol)
	else:
		return {'error': 'Unknown source'}, 400
	return data

if __name__ == '__main__':
	app.run(debug=True)
