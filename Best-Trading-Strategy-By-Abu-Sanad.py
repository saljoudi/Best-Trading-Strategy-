import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import yfinance as yf
import ta
import plotly.graph_objs as go
from itertools import product
import time
from tqdm import tqdm

# Initialize the Dash app with a Bootstrap theme for a professional look
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
server = app.server  # Expose the Flask server

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Best Trading Strategy Finder", className="text-center"), className="mb-4 mt-4")
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Input Parameters", className="card-title"),
                    dbc.Label("Ticker Symbol (without .SR for Saudi stocks):"),
                    dcc.Input(id='ticker-input', type='text', value='1303', className="mb-3", style={'width': '100%'}),
                    dbc.Label("Period:"),
                    dcc.Dropdown(
                        id='period-input',
                        options=[
                            {'label': '1 Year', 'value': '1y'},
                            {'label': '2 Year', 'value': '2y'},
                            {'label': '3 Year', 'value': '3y'},
                            {'label': 'All', 'value': 'max'}
                        ],
                        value='1y',
                        className="mb-3",
                        style={'width': '100%'}
                    ),
                    dbc.Button("Analyze", id="analyze-button", color="primary", className="mt-3", style={'width': '100%'})
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Progress", className="card-title"),
                    dbc.Progress(id="progress-bar", striped=True, animated=True, color="success", className="mb-3"),
                    html.H4("Strategy Summary", className="card-title"),
                    html.Pre(id='summary-output', style={'whiteSpace': 'pre-wrap', 'font-family': 'monospace'})
                ])
            ])
        ], width=9)
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Trades Details", className="card-title"),
                    html.Div(id='trades-table')
                ])
            ])
        ], width=12)
    ])
], fluid=True)


@app.callback(
    [Output('summary-output', 'children'),
     Output('trades-table', 'children'),
     Output('progress-bar', 'value')],
    [Input('analyze-button', 'n_clicks')],
    [Input('ticker-input', 'value'),
     Input('period-input', 'value')]
)
def perform_grid_search(n_clicks, ticker_input, period):
    if n_clicks is None:
        return "", "", 0

    # Check if the ticker is numeric (Saudi stock symbol)
    if ticker_input.isdigit():
        ticker = f"{ticker_input}.SR"
    else:
        ticker = ticker_input

    # Download the data for the ticker
    df = yf.download(ticker, period=period)

    # Define parameter ranges for grid search
    sma_short_range = range(5, 20, 2)  # Short SMA periods from 5 to 18
    sma_long_range = range(10, 50, 5)  # Long SMA periods from 20 to 45
    rsi_threshold_range = range(40, 55, 5)  # RSI thresholds from 40 to 50
    adl_short_range = range(5, 20, 2)  # Short ADL SMA periods from 5 to 18
    adl_long_range = range(10, 50, 5)  # Long ADL SMA periods from 20 to 45

    # Generate all possible combinations of parameters
    parameter_grid = list(product(sma_short_range, sma_long_range, rsi_threshold_range, adl_short_range, adl_long_range))
    total_combinations = len(parameter_grid)
    best_params = None
    best_performance = float('-inf')
    results = {}

    progress = 0

    for params in tqdm(parameter_grid):
        sma_short, sma_long, rsi_threshold, adl_short, adl_long = params

        # Calculate SMAs and ADL SMAs with the current parameters
        df['SMA_Short'] = df['Close'].rolling(window=sma_short).mean()
        df['SMA_Long'] = df['Close'].rolling(window=sma_long).mean()
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        df['MACD'] = ta.trend.MACD(df['Close']).macd()
        df['MACD_Signal'] = ta.trend.MACD(df['Close']).macd_signal()
        df['ADL'] = ta.volume.AccDistIndexIndicator(df['High'], df['Low'], df['Close'], df['Volume']).acc_dist_index()
        df['ADL_Short_SMA'] = df['ADL'].rolling(window=adl_short).mean()
        df['ADL_Long_SMA'] = df['ADL'].rolling(window=adl_long).mean()

        # Signal generation based on the current parameters
        df['Signal'] = 0  # Default to no signal
        df['Signal'] = df.apply(
            lambda row: 1 if row['SMA_Short'] > row['SMA_Long'] and row['ADL_Short_SMA'] > row['ADL_Long_SMA'] and row['RSI'] >= rsi_threshold and row['MACD'] > row['MACD_Signal'] else (
                -1 if row['SMA_Short'] < row['SMA_Long'] and row['ADL_Short_SMA'] < row['ADL_Long_SMA'] and row['RSI'] < rsi_threshold and row['MACD'] <= row['MACD_Signal'] else 0
            ), axis=1
        )

        # Simulate trading with the generated signals
        initial_investment = 100000  # Example initial investment
        portfolio = initial_investment
        trades = []
        buy_price = None
        trade_start = None
        number_of_trades = 0

        for index, row in df.iterrows():
            if row['Signal'] == 1 and buy_price is None:
                buy_price = row['Close']
                trade_start = index
                number_of_trades += 1
            elif row['Signal'] == -1 and buy_price is not None:
                sell_price = row['Close']
                profit = (sell_price - buy_price) * (portfolio / buy_price)
                portfolio += profit
                days_held = (index - trade_start).days

                trades.append({
                    'Sell Date': index.date().strftime('%Y-%m-%d'),
                    'Buy Price': f"{buy_price:.2f} SAR",
                    'Sell Price': f"{sell_price:.2f} SAR",
                    'Days Held': days_held,
                    'Profit': f"{profit:,.2f} SAR",
                    'Profit Percentage': f"{(profit / (portfolio - profit)) * 100:.2f}%"
                })

                buy_price = None  # Reset after trade

        final_value = portfolio
        total_return = final_value - initial_investment
        percentage_return = (total_return / initial_investment) * 100

        # Store results
        results[params] = {
            'Initial Investment': initial_investment,
            'Final Portfolio Value': final_value,
            'Total Return': total_return,
            'Percentage Return': percentage_return,
            'Number of Trades': number_of_trades,
            'Average Days Held per Trade': sum([t['Days Held'] for t in trades]) / number_of_trades if number_of_trades > 0 else 0,
            'Sell Trades': trades
        }

        # Check if this combination is the best so far
        if percentage_return > best_performance:
            best_performance = percentage_return
            best_params = params

        # Update progress
        progress += 1
        progress_percent = int((progress / total_combinations) * 100)
        time.sleep(0.1)  # Simulate computation delay

    # Best combination results
    summary_text = (
        f"Best Parameters: SMA_Short={best_params[0]}, SMA_Long={best_params[1]}, RSI_Threshold={best_params[2]}, ADL_Short_SMA={best_params[3]}, ADL_Long_SMA={best_params[4]}\n"
        f"Initial Investment: 100,000 SAR\n"
        f"Final Portfolio Value: {results[best_params]['Final Portfolio Value']:,.2f} SAR\n"
        f"Total Return: {results[best_params]['Total Return']:,.2f} SAR\n"
        f"Percentage Return: {results[best_params]['Percentage Return']:.2f}%\n"
        f"Number of Trades: {results[best_params]['Number of Trades']}\n"
        f"Average Days Held per Trade: {results[best_params]['Average Days Held per Trade']:.2f} days"
    )


    # Create the trades table
    trades_df = pd.DataFrame(results[best_params]['Sell Trades'])
    trades_table = dbc.Table.from_dataframe(trades_df, striped=True, bordered=True, hover=True)

    return summary_text, trades_table, progress_percent


if __name__ == '__main__':
    app.run_server(debug=True)
