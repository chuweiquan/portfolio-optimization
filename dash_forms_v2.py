import dash
import dash_bootstrap_components as dbc
import numpy
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import statsmodels.api as sm
import scipy.stats as stat

from dash.exceptions import PreventUpdate
from dash import html
from dash import dcc
from dash import dash_table
from dash.dependencies import Input, Output, State
from scipy.stats.stats import skew
from scipy.stats import norm
from pandas_datareader import data
from plotly.subplots import make_subplots 
from numpy import array
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from datetime import timedelta, date


from datetime import datetime
today = datetime.today().strftime('%Y-%m-%d')

TICKERS = [
    'BTC-USD','ETH-USD','BNB-USD','USDT-USD','ADA-USD','SOL1-USD','XRP-USD','DOT1-USD','HEX-USD','SHIB-USD','DOGE-USD','USDC-USD','LUNA1-USD','UNI3-USD','AVAX-USD','LINK-USD','LTC-USD','MATIC-USD','ALGO-USD','BCH-USD','XLM-USD','AXS-USD','VET-USD','ATOM1-USD','ICP1-USD','THETA-USD','TRX-USD','FIL-USD','ETC-USD','FTT1-USD','FTM-USD','DAI1-USD','MANA-USD','HBAR-USD','XTZ-USD','CRO-USD','EGLD-USD','XMR-USD','EOS-USD','FLOW1-USD','GRT2-USD','CAKE-USD','AAVE-USD','MIOTA-USD','RUNE-USD','QNT-USD','ONE2-USD','BSV-USD','KSM-USD','NEO-USD','HNT1-USD','CHZ-USD','WAVES-USD','BTT1-USD','MKR-USD','STX1-USD','ENJ-USD','ZEC-USD','CELO-USD','DASH-USD','COMP-USD','AMP1-USD','TFUEL-USD','CRV-USD','OMG-USD','HOT1-USD','BAT-USD','AR-USD','SAND-USD','XEM-USD','DCR-USD','CTC1-USD','ICX-USD','SUSHI-USD','ZIL-USD','ZEN-USD','QTUM-USD','YFI-USD','ANKR-USD','TUSD-USD','RVN-USD','SNX-USD','BTG-USD','XDC-USD','CEL-USD','ZRX-USD','OMI-USD','CCXX-USD','BNT-USD','SC-USD','SRM-USD','KDA-USD','ONT-USD','IOST-USD','1INCH-USD','NANO-USD','WAXP-USD','RAY-USD','LRC-USD','VGX-USD','LRC-USD','DGB-USD','UMA-USD','CELR-USD','WIN1-USD','GNO-USD','C98-USD','XWC-USD','IOTX-USD','DFI-USD','NU-USD','FET-USD','GLM-USD','CKB-USD','KAVA-USD','STORJ-USD','RSR-USD','COTI-USD','SXP-USD','LSK-USD','NMR-USD','VTHO-USD','XVG-USD','MED-USD','TWT-USD','BCD-USD','VLX-USD','CTSI-USD','ARRR-USD','RLC-USD','SNT-USD','CVC-USD','VRA-USD','ARDR-USD','BAND-USD','HIVE-USD','ERG-USD','NKN-USD','STMX-USD','ETN-USD','EWT-USD','OXT-USD','ROSE-USD','STRAX-USD','REP-USD','SAPP-USD','ARK-USD','DAG-USD','MIR1-USD','MLN-USD','MAID-USD','STEEM-USD','XCH-USD','TOMO-USD','FUN-USD','MTL-USD','DERO-USD','ZNN-USD','SYS-USD','ACH-USD','PHA-USD','ANT-USD','WAN-USD','BAL-USD','RBTC-USD','CLV-USD','AVA-USD','META-USD','KIN-USD','BTS-USD','ADX-USD','KMD-USD','MCO-USD','IRIS-USD','HNS-USD','XHV-USD','NYE-USD','FIRO-USD','TT-USD','ZEL-USD','ABBC-USD','DNT-USD','MONA-USD','XNC-USD','NRG-USD','ELA-USD','GAS-USD','AION-USD','DMCH-USD','DIVI-USD','WOZX-USD','BTM-USD','PAC-USD','BEPRO-USD','NIM-USD','GRS-USD','FRONT-USD','WTC-USD','REV-USD','APL-USD','BEAM-USD','CUDOS-USD','FIO-USD','BCN-USD','DGD-USD','SBD-USD','VERI-USD','RDD-USD','SRK-USD','NULS-USD','VITE-USD','PCX-USD','MARO-USD','XCP-USD','SOLVE-USD','PIVX-USD','SERO-USD','AXEL-USD','CET-USD','NXS-USD','VSYS-USD','ATRI-USD','GXC-USD','VTC-USD','CRU-USD','CUT-USD','AE-USD','MWC-USD','GO-USD','FSN-USD','CTXC-USD','ADK-USD','GRIN-USD','KRT-USD','LOKI-USD','ZANO-USD','WICC-USD','PPT-USD','MHC-USD','GBYTE-USD','NAV-USD','MASS-USD','QASH-USD','VAL1-USD','NEBL-USD','XSN-USD','GAME-USD','NMC-USD','HC-USD','NAS-USD','BTC2-USD','ETP-USD','AMB-USD','PPC-USD','RSTR-USD','LBC-USD','PAI-USD','FO-USD','WABI-USD','PART-USD','NXT-USD','CHI-USD','BIP-USD','SALT-USD','MAN-USD','DTEP-USD','QRL-USD','SKY-USD','OBSR-USD','FCT-USD','PI-USD','MRX-USD','DCN-USD','TRUE-USD','PZM-USD','DMD-USD','EMC2-USD','LCC-USD','BHP-USD','PLC-USD','RINGX-USD','INSTAR-USD','TRTL-USD','QRK-USD','PAY-USD','YOYOW-USD','HPB-USD','SCC3-USD','SCP-USD','UBQ-USD','LEDU-USD','NLG-USD','DNA1-USD','NVT-USD','ACT-USD','XDN-USD','BHD-USD','BLOCK-USD','SFT-USD','SMART-USD','POA-USD','CMT1-USD','HTML-USD','AEON-USD','XMY-USD','WGR-USD','GLEEC-USD','INT-USD','DYN-USD','VIA-USD','XMC-USD','VEX-USD','GHOST1-USD','IDNA-USD','FLO-USD','ZYN-USD','PMEER-USD','FTC-USD','HTDF-USD','BTX-USD','TERA-USD','VIN-USD','OTO-USD','BLK-USD','ILC-USD','CURE-USD','WINGS-USD','NYZO-USD','MIR-USD','EDG-USD','GRC-USD','XST-USD','USNBT-USD','IOC-USD','GCC1-USD','DIME-USD','POLIS-USD','FTX-USD','COLX-USD','CRW-USD','BCA-USD','OWC-USD','FAIR-USD','TUBE-USD','SONO1-USD','PHR-USD','MBC-USD','XLT-USD','SUB-USD','AYA-USD','GHOST-USD','BPS-USD','NIX-USD','XRC-USD','MGO-USD','XBY-USD','DDK-USD','ERK-USD','HYC-USD','XAS-USD','BPC-USD','SNGLS-USD','ATB-USD','FRST-USD','COMP1-USD','OURO-USD','BDX-USD','ALIAS-USD','FLASH-USD','NLC2-USD','CSC-USD','ECC-USD','CLAM-USD','UNO-USD','BONO-USD','MOAC-USD','LKK-USD','ECA-USD','DACC-USD','RBY-USD','HNC-USD','SPHR-USD','MINT-USD','AIB-USD','XUC-USD','HONEY3-USD','DUN-USD','MTC2-USD','JDC-USD','CCA-USD','SLS-USD','DCY-USD','MIDAS-USD','BRC-USD','GRN-USD','KNC-USD','LRG-USD','BONFIRE-USD','BST-USD','SFMS-USD','BST-USD'
]

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(
    [
        # inputs section
        html.Div(
            dbc.Container(
                [
                    html.H1("Input the following information", className="display-3"),
                    html.Hr(className="my-2"),
                    # this section is for input investment amount
                    # dbc.Row(
                    #     [
                    #         dbc.Col(
                    #             [
                    #                 dbc.Label("Investment Amount", html_for="investment"),
                    #                 dbc.Input(
                    #                     type="text",
                    #                     id="investment",
                    #                 ),
                    #             ],
                    #         ),
                    #     ]
                    # ),
                    # this section is for the input start and end date
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Select the start date you would like to analyze", html_for="start_analyze_date"),
                                    dbc.Input(
                                        type="date",
                                        id="start_analyze_date",
                                        value="2020-01-01"
                                    ),
                                ],
                                width=6, style={'padding': 10},
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Select the end date you would like to analyze", html_for="end_analyze_date"),
                                    dbc.Input(
                                        type="date",
                                        id="end_analyze_date",
                                        value="2020-12-31"
                                    ),
                                ],
                                width=6, style={'padding': 10},
                            ),   
                        ]
                    ),
                    # get data information
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Button("Confirm", className="mr-1 align-self-end", color="light", id="pull_data", n_clicks=0 ),
                                    # dbc.Button("Start Building Portfolio", className="mr-1", id="get_reco_portf", n_clicks=0 ),
                                ], style={'padding': 10},
                            ),
                        ]
                    )
                ],
                fluid=True,
                className="py-3",
            ),
            className="p-3 text-white bg-dark rounded-3",
        ),
        dcc.Store(id="store_all_tickers"),
        dcc.Store(id="store_ticker_dataframe"),
        dcc.Store(id="mpt_dataframe"),
        dcc.Store(id="top_10_tickers"),
        # start analysis section
        html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label("Select the crypto you would like it analyze", html_for="crypto_dropdown"),
                                dcc.Dropdown(
                                        id="crypto_dropdown",
                                        # options=[{"value": x, "label": x} for x in TICKERS]
                                    ),
                            ],
                            width=4,
                            style={
                                'padding-top': 50,
                                'padding-bottom': 10
                            },
                        ),
                    ],
                ),
                dbc.Button("Generate Analysis Charts", color="dark", className="mr-1", id="submit_analysis_charts", n_clicks=0 ),
                # pin1
                # this section displays the analysis charts
                dbc.Row(id="display_statistics", className="g-0"),
                html.Div(id="display_charts",
                    className="g-0",
                    style={
                        'padding-top': 50,
                        'padding-bottom': 100
                    },
                ),
            ],
        ),
        # MPT Toggle Section (skip first come back to this later)
        html.Div(
            [
                dbc.Button("Toggle View MPT Chart", color="dark", className="mr-1", id="toggle_mpt", n_clicks=0 ),
                dbc.Spinner(
                    html.Div(
                        id="mpt_chart",
                        style={
                            'padding-bottom': 100
                        },
                    ),
                ),
            ]
        ),
        # selecting gut feel portfolio and recommended portfolio
        dbc.Row(
            [   
                # this section adds their gut feel portfolio
                dbc.Col(
                    [
                        html.H3(["Select Your Portfolio"], className="text-center"),
                        dbc.Row(
                            [
                                html.Div(
                                    [
                                        dbc.Label("Select crypto", html_for="dropdown"),
                                        dcc.Dropdown(
                                            id="gut_feel_dropdown",
                                        ),
                                    ],
                                    className="col-4"
                                ),
                                html.Div(
                                    [
                                        dbc.Label("Portfolio Weight", html_for="portfolio_weight"),
                                        dbc.Input(
                                            id="portfolio_weight",
                                            type="number"
                                        ),
                                    ],
                                    className="col-4"
                                ),
                                html.Div(
                                    [
                                        dbc.Button("Add Crypto", color="dark", className="mt-1 align-self-end", id="submit_crypto", n_clicks=0),
                                    ],
                                    className="col-4 d-flex"
                                ),
                            ],
                            style={
                                'padding-top': 10,
                                'padding-bottom': 10,
                            }
                        ),
                        dbc.Row(
                            [
                                html.Div(id="gut_feel_portfolio_metrics"),
                                html.Div(
                                    [
                                        dbc.Alert(
                                            id="alert_auto",
                                            is_open=False,
                                            duration=2000,
                                            style={
                                                'padding-top': 20,
                                                'padding-bottom': 20
                                            }
                                        ),
                                        dash_table.DataTable(
                                            id='portfolio_table',
                                            columns=[{'name': 'Crypto', 'id': 'crypto', 'presentation': 'dropdown', 'deletable': False, 'renamable': False},
                                                    {'name': 'weightage', 'id': 'weightage', 'deletable': False, 'renamable': False}
                                            ],
                                            data=[{'crypto': 'Select your crypto ticker from the dropdown', 'weightage': 'Input your desired weights for the crypto'},
                                            ],
                                            # editable=True,                  # allow user to edit data inside tabel
                                            row_deletable=True,             # allow user to delete rows
                                            # sort_action="native",           # give user capability to sort columns
                                            sort_mode="single",             # sort across 'multi' or 'single' columns
                                            # filter_action="native",         # allow filtering of columns
                                            page_action='none',             # render all of the data at once. No paging.
                                            # style_table={'height': '200px', 'overflowY': 'auto'},
                                            style_cell={'textAlign': 'center'},
                                            style_data={
                                                'whiteSpace': 'normal',
                                                'height': 'auto',
                                            },
                                        ),   
                                    ],
                                ),
                            ],
                            style={
                                'padding-top': 20,
                                'padding-bottom': 30,
                                'text-align': 'center'
                            },
                        ),
                    ],
                    width=6,
                ),
                dbc.Col(
                    [
                        html.H3(["Our Recommended Portfolio"], className="text-center"),
                        # this section switches the views of recommended portfolio
                        dbc.Tabs(
                            [
                                # tab 1
                                dbc.Tab(
                                    label="Best Portfolio",
                                    tab_id = "best"
                                ),
                                # tab 2
                                dbc.Tab(
                                    label="Adjust Volatility",
                                    tab_id = "volatility"
                                ),
                                # tab 3
                                dbc.Tab(
                                    label="Adjust Returns",
                                    tab_id = "returns"
                                ),
                            ],
                            id="tabs",
                            active_tab = "best"
                        ),
                        dbc.Spinner(
                            dbc.Row(
                                [   
                                    html.Div(
                                        [
                                            dbc.Label(id="vol_rtn_label", html_for="vol_rtn_input", align="start", style={
                                                "padding": 10
                                            }),
                                            dbc.Input(
                                                type="number",
                                                id="vol_rtn_input",
                                                min = 0,
                                                max = 0,
                                                value = 0
                                            ),
                                        ],
                                        className = "col-11"
                                    ),
                                    html.Div(
                                        [
                                            dbc.Button("Send", color="dark", className="mt-1 align-self-end", id="submit_vol_rtn_input", n_clicks=0)
                                        ],
                                        className = "col-1 d-flex"
                                    ),

                                    html.Div(id="portfolio_metrics"),
                                    html.Div(
                                        [
                                            dbc.Alert(
                                                id="recommended_alert_auto",
                                                is_open=False,
                                                duration=2000,
                                                style={
                                                    'padding-top': 20,
                                                    'padding-bottom': 20
                                                }
                                            ),
                                            dash_table.DataTable(
                                                id='portfolio_tables',
                                                columns=[{'name': 'Crypto', 'id': 'crypto', 'presentation': 'dropdown', 'deletable': False, 'renamable': False},
                                                        {'name': 'weightage', 'id': 'weightage', 'deletable': False, 'renamable': False}
                                                ],
                                                data=[],
                                                # editable=True,                  # allow user to edit data inside tabel
                                                # row_deletable=True,             # allow user to delete rows
                                                # sort_action="native",           # give user capability to sort columns
                                                sort_mode="single",             # sort across 'multi' or 'single' columns
                                                # filter_action="native",         # allow filtering of columns
                                                page_action='none',             # render all of the data at once. No paging.
                                                # style_table={'height': '200px', 'overflowY': 'auto'},
                                                style_cell={'textAlign': 'center'},
                                                style_data={
                                                    'whiteSpace': 'normal',
                                                    'height': 'auto',
                                                },
                                            ),   
                                        ],
                                    ),
                                ],
                                style={
                                    'text-align': 'center',
                                    'padding-top': 20
                                },
                            )
                        )
                    ]
                )
            ],
            style={
                'padding-bottom': 100
            },
        ),
        dbc.Button("Generate Portfolio Comparison", color="dark", className="mr-1", id="submit_comparison", n_clicks=0),
        
        dbc.Row(
            [
                dbc.Alert(id="portf_comparison_alert", is_open=False, duration=2000,
                    style={
                        'padding-top': 20,
                        'padding-bottom': 20
                    }
                ),

                # weightage comparison
                dbc.Spinner(
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Div(
                                    [],
                                    id="gut_feel_weights"
                                ),
                                width=6,
                            ),
                            dbc.Col(
                                html.Div(
                                    [],
                                    id="recommended_weights"
                                ),
                                width=6,
                            ),
                        ]
                    ),
                ),
                
                # returns comparison
                dbc.Spinner(
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Div(id="gut_feel_returns"),
                                width=6,
                            ),
                            dbc.Col(
                                html.Div(id="recommended_returns"),
                                width=6,
                            ),
                        ]
                    ),
                ),

                # sharpe ratio comparison
                dbc.Spinner(
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Div(id="gut_feel_vol"),
                                width=6,
                            ),
                            dbc.Col(
                                html.Div(id="recommended_vol"),
                                width=6,
                            ),
                        ]
                    ),
                ),
            ],
            style={
                'padding-bottom': 100,
                'padding-top': 20
            },
        ),
        

        dbc.Button("Generate LSTM Predictions", color="dark", className="mr-1", id="lstm_button", n_clicks=0),
        dbc.Row(
            # pin2
            [
                html.Div(id="lstm_graph"),
                html.Div(id="lstm_table"),
                
            ],
            style={
                'padding-bottom': 100
            },
        )
    ]
)

# callbacks that control storage of data
@app.callback(
    [Output("store_all_tickers", "data"),
    Output("store_ticker_dataframe", "data"),
    Output("mpt_dataframe", "data"),
    Output("top_10_tickers", "data")],
    Input("pull_data", "n_clicks"),
    [State("start_analyze_date", "value"),
    State("end_analyze_date", "value")]
)
def pull_data(clicks, start_date, end_date):
    prices_df = yf.download(TICKERS, start=start_date, end=end_date, adjusted=True)
    vol_df = prices_df['Volume']

    # begin getting top10 here
    mpt_df, top_10_crypto, tickers_dropdown = get_mpt_df(prices_df)

    tickers = {"tickers dropdown": tickers_dropdown}
    ticker_dataframe = vol_df.to_dict('records')
    ticker_mpt_df = mpt_df.to_dict('records')
    return tickers, ticker_dataframe, ticker_mpt_df, top_10_crypto

def get_mpt_df(prices_df):
    vol_df = prices_df['Volume']
    all_close_df = prices_df['Close']
    nan_value = float("NaN")
    vol_df.replace(0, nan_value, inplace=True)
    vol_df = vol_df.dropna(1)
    dropdown_tickers = vol_df.columns.tolist()
    
    mean_vol = vol_df.mean(axis=0)
    mean_vol_df = pd.DataFrame(mean_vol, columns = ['Average Volume'])

    top100_avg_vol = mean_vol_df.nlargest(100, 'Average Volume')
    tickers = top100_avg_vol.index.values

    sharpe_ratio_set = {"ticker" : (tickers) , "Volatility" : ([]),'Annualised Returns' : ([]),'Sharpe Ratio' : ([])}

    risk_free_rate = 0
    years = 1

    for ticker in tickers:
        # panel_data = yf.download(ticker, start=start_date, end=end_date, adjusted=True)
        
        #close price series
        close_df = all_close_df[ticker]
        
        #calculate close return series
        close_pct_change = close_df.pct_change()
        close_return_series = ((1 + close_pct_change).cumprod() - 1)
        close_returns_df = close_return_series.to_frame()

        #calculate annualised returns
        annualized_returns_df = (1 + close_returns_df.tail(1))**(1/years)-1
        annualised_returns = annualized_returns_df.iloc[0][0]
        sharpe_ratio_set['Annualised Returns'].append(annualised_returns)
        
        #calculate volatility
        volatility = np.sqrt(np.log(close_df/close_df.shift(1)).var()) * np.sqrt(365)
        sharpe_ratio_set['Volatility'].append(volatility)

        #calculate annualised historical volatility
        annualised_historical_volatility = np.sqrt(365) * pd.DataFrame.rolling(np.log(close_df / close_df.shift(1)),window=20).std()
    
        #calculate sharpe ratio
        returns_ts = close_pct_change.dropna()
        returns_ts = returns_ts.to_frame()
        returns_ts.rename(columns={ticker: "Close"},inplace = True)
        avg_daily_returns = returns_ts['Close'].mean()
        returns_ts['Risk Free Rate'] = risk_free_rate/365
        avg_rfr_ret = returns_ts['Risk Free Rate'].mean()
        returns_ts['Excess Returns'] = returns_ts['Close'] - returns_ts['Risk Free Rate']
        sharpe_ratio = ((avg_daily_returns - avg_rfr_ret) /(returns_ts['Excess Returns'].std()))*np.sqrt(365)
        sharpe_ratio_set['Sharpe Ratio'].append(sharpe_ratio) 

    sharpe_ratio_df = pd.DataFrame(sharpe_ratio_set).sort_values(by='Sharpe Ratio',ascending=False)
    positive_sharpe_ratio_df = sharpe_ratio_df[sharpe_ratio_df['Sharpe Ratio'] > 0]
    positive_sharpe_ratio_df = positive_sharpe_ratio_df.reset_index()

    first_ticker = positive_sharpe_ratio_df['ticker'].values[0]
    top_list = []
    for i in positive_sharpe_ratio_df['ticker'].values:
        top_list.append(i)

    confirmed_top_list = top_list
    # top_df = yf.download(top_list, start=start_date, end=end_date, adjusted=True)
    #get return series of top 10 sharpe ratio
    top_close = all_close_df[top_list]
    close_pct_change = top_close.pct_change()
    close_return_df = ((1 + close_pct_change).cumprod() - 1)
    close_return_df.fillna(0,inplace = True)
    corr_matrix = close_return_df.corr()
    n = (len(top_list)*(len(top_list)-1))/2

    top_pairs = get_top_abs_correlations(close_return_df, int(n))
    top_pairs = pd.DataFrame(top_pairs, columns = ['Correlation'])
    removed_top_pairs = top_pairs[top_pairs['Correlation']>0.9]
    removed_top_pairs = removed_top_pairs.index

    removed_list = []
    confirmed_list = []
    for ticker in top_list:
        for i in removed_top_pairs:
            ticker1 = i[0]
            ticker2 = i[1]
            if ticker == ticker1 and ticker2 not in removed_list:
                removed_list.append(ticker2)
            if ticker == ticker2 and ticker1 not in removed_list:
                removed_list.append(ticker1)
        if ticker not in removed_list:
            confirmed_list.append(ticker)
        top_list = [crypto for crypto in top_list if crypto not in removed_list] 
        top_list = [crypto for crypto in top_list if crypto not in confirmed_list] 

    if confirmed_top_list[0] not in confirmed_list:
        confirmed_list = list(confirmed_top_list[0]) + confirmed_list

    final_list = confirmed_list[:10]
    returns_df = all_close_df[final_list]

    n_portfolios = 10 ** 5
    n_assets = len(final_list)
    

    returns_series_df = returns_df.pct_change().dropna()

    avg_returns = returns_series_df.mean() * 365
    cov_mat = returns_series_df.cov() * 365

    # Simulate random portfolio weights
    np.random.seed(42)
    weights = np.random.random(size=(n_portfolios, n_assets))
    weights /=  np.sum(weights, axis=1)[:, np.newaxis]

    # Calculate portfolio metrics
    portf_rtns = np.dot(weights, avg_returns)

    portf_vol = []
    for i in range(0, len(weights)):
        portf_vol.append(np.sqrt(np.dot(weights[i].T, np.dot(cov_mat, weights[i]))))

    portf_vol = np.array(portf_vol)  

    portf_sharpe_ratio = portf_rtns / portf_vol

    # Create a joint DataFrame with all data
    portf_results_df = pd.DataFrame({'returns': portf_rtns,
                                    'volatility': portf_vol,
                                    'sharpe_ratio': portf_sharpe_ratio})


    return portf_results_df, final_list, dropdown_tickers

def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=35):
    au_corr = df.corr().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

# callbacks that controll all the dropdown options
@app.callback(
    [Output("crypto_dropdown", "options"),
    Output("gut_feel_dropdown", "options")],
    Input("store_all_tickers", "data"),
    prevent_initial_call = True
)
def display_dropdown(data):
    options = [{"value": x, "label": x} for x in data["tickers dropdown"]]
    return options, options

# callbacks that control analysis charts for each ticker
@app.callback(
    [Output('display_statistics', 'children'),
    Output('display_charts', 'children')],
    Input('submit_analysis_charts', 'n_clicks'),
    [State('crypto_dropdown', 'value'),
    State('start_analyze_date', 'value'),
    State('end_analyze_date', 'value')],
    prevent_initial_call = True
)
def update_charts(clicks, crypto, start_date, end_date):
    # receiving chart data
    ticker = crypto
    years = 1
    close_pct_change, close_return_series, ahv, volatility, sharpe_ratio, annualized_returns, statistics, skewness, kurtosis = all_funcs(ticker, years, start_date, end_date)

    stats_dict = {
        "Volatility": volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Annualized Returns": annualized_returns,
        "Skewness": skewness,
        "Kurtosis": kurtosis,
    }

    # round values in stats_dict to 2dp
    for key,value in stats_dict.items():
        stats_dict[key] = round(value, 2)

    statistics_df = pd.DataFrame(stats_dict)
    # statistics_df = statistics_df.reset_index()

    close_pct_change = pd.DataFrame(close_pct_change)
    close_pct_change = close_pct_change.reset_index()

    close_return_series =  pd.DataFrame(close_return_series)
    close_return_series = close_return_series.reset_index()

    ahv = pd.DataFrame(ahv)
    ahv = ahv.reset_index()
    
    df1 = close_pct_change
    print(df1.head())
    print(close_return_series.head())

    display_stats = html.Div(
                        [
                            html.H3(["Crypto Statistics"], className="text-center"),
                            dbc.Table.from_dataframe(statistics_df, striped=True, bordered=True, hover=True),
                        ]
                    ),
    display_charts = dbc.Row(
                        [
                            html.H3(["Crypto Charts"], className="text-center"),
                            dbc.Col(
                                dcc.Graph(
                                    figure = {
                                        'data': [
                                            {'x': df1['Date'], 'y': df1['Close'],'type':'line'}
                                        ],
                                        'layout' : {
                                            'title': str(crypto) + ' Close Percentage Change'
                                        }
                                    }
                                ),
                            ),
                            dbc.Col(
                                dcc.Graph(
                                    figure = {
                                        'data': [
                                            {'x': close_return_series['Date'], 'y': close_return_series['Close'],'type':'line'}
                                        ],
                                        'layout' : {
                                            'title': str(crypto) + ' Close Return Series'
                                        }
                                    }
                                ),
                            ),
                            dbc.Col(
                                dcc.Graph(
                                    figure = {
                                        # pin1.1
                                        'data': [
                                            {'x': ahv['Date'], 'y': ahv['Close'],'type':'line'}
                                        ],
                                        'layout' : {
                                            'title': str(crypto) + ' Historical Rolling Volatility'
                                        }
                                    }
                                )
                            )
                        ]
                    )

    return display_stats, display_charts

def load_data(ticker, start_date, end_date):
    crypto_data = yf.download(ticker, start=start_date, end=end_date, adjusted=True)

    return crypto_data['Close']

def chart_analysis(crypto, years):
    statistics = crypto.describe()

    # percentage change
    close_pct_change = crypto.pct_change()
    
    skewness = close_pct_change.skew()
    kurtosis = close_pct_change.kurtosis()
    
    # calculating return series
    close_return_series = (1 + close_pct_change).cumprod() - 1
    annualized_returns = (1 + close_return_series.tail(1))**(1/years)-1
    
    # calculating annual volatility
    volatility = np.sqrt(np.log(crypto / crypto.shift(1)).var()) * np.sqrt(365)
    ahv = np.sqrt(252) * pd.DataFrame.rolling(np.log(crypto / crypto.shift(1)),window=20).std()
    
    # calculating sharpe ratio
    risk_free_rate = 0
    returns_ts = close_pct_change.dropna()
    avg_daily_returns = returns_ts.mean()
    
    returns_ts['Risk Free Rate'] = risk_free_rate/252
    avg_rfr_ret = returns_ts['Risk Free Rate'].mean()
    returns_ts['Excess Returns'] = returns_ts - returns_ts['Risk Free Rate']

    sharpe_ratio = ((avg_daily_returns - avg_rfr_ret) /returns_ts['Excess Returns'].std())*np.sqrt(365)

    return close_pct_change, close_return_series, ahv, volatility, sharpe_ratio, annualized_returns, statistics, skewness, kurtosis

def all_funcs(ticker, years, start_date, end_date):
    crypto = load_data(ticker, start_date, end_date)
    return chart_analysis(crypto, years)

# this section controls callbacks for mpt chart
@app.callback(
    Output("mpt_chart", "children"),
    Input("toggle_mpt", "n_clicks"),
    State("mpt_dataframe", "data")
)
def display_mpt(n_clicks, data):
    if n_clicks % 2 != 0:
        df = pd.DataFrame(data)

        fig = dcc.Graph(
            figure = px.scatter(df, x="volatility", y="returns", color='sharpe_ratio')
        ) 
        
    else:
        fig = html.Div()

    return fig

# this section controls callbacks for gut feel portfolio
@app.callback(
    [Output("portfolio_table", 'data'),
    Output("alert_auto", 'children'),
    Output("alert_auto", 'color'),
    Output("alert_auto", 'is_open'),
    Output("gut_feel_portfolio_metrics", 'children')],
    Input('submit_crypto', 'n_clicks'),
    [State('gut_feel_dropdown', 'value'),
    State('portfolio_weight', 'value'),
    State('portfolio_table', 'data'),
    State('alert_auto', 'is_open'),
    State('start_analyze_date', 'value'),
    State('end_analyze_date', 'value')],
    prevent_initial_call = True
)
def add_row(n_clicks, crypto, weight, rows, is_open, start_date, end_date):
    cryptos_list = []
    weights_list = []
    for i in range(len(rows)):
        cryptos_list.append(rows[i]["crypto"])
        weights_list.append(rows[i]["weightage"])

    if isinstance(weights_list[0], str):
        if crypto == None:
            text = "Please select your crypto"
            color = "danger"
        
        elif weight == None:
            text = "Please enter your weights for the crypto"
            color = "danger"

        elif weight > 1:
            text = "Your total weightage is more than 1"
            color = "danger"

        else:
            rows = [{'crypto': crypto, 'weightage': weight}]
            cryptos_list = [crypto]
            weights_list = [weight]
            text = "Your crypto has been added"
            color = "success"

        is_open = True

    else:
        if crypto == None:
            text = "Please select your crypto"
            color = "danger"   

        elif weight == None:
            text = "Please enter your weights for the crypto"
            color = "danger"

        elif len(rows) < 10:
            if crypto in cryptos_list:
                text = "You have added this crypto before"
                color = "danger"

            elif (isinstance(weights_list[0], float) or isinstance(weights_list[0], int)) and sum(weights_list) + weight > 1:
                text = "Your total weightage is more than 1"
                color = "danger"

            else:
                rows.append({'crypto': crypto, 'weightage': weight})
                cryptos_list.append(crypto)
                weights_list.append(weight)
                text = "Your crypto has been added"
                color = "success"

        is_open = True

    # this section generates the sharpe, returns and vol table
    if isinstance(weights_list[0], float) and round(sum(weights_list), 1) == 1.0:
        returns, volatility, sharpe_ratio = get_metrics(cryptos_list, weights_list, start_date, end_date)
        
        recommended_metrics = pd.DataFrame({"Returns": [returns], "Volatility": [volatility], "Sharpe Ratio": [sharpe_ratio]})

        fig = dbc.Table.from_dataframe(recommended_metrics, striped=True, bordered=True, hover=True)
    
    else:
        recommended_metrics = pd.DataFrame({"Returns": ["-"], "Volatility": ["-"], "Sharpe Ratio": ["-"]})

        fig = dbc.Table.from_dataframe(recommended_metrics, striped=True, bordered=True, hover=True)

    return rows, text, color, is_open, fig

def get_metrics(tickers, weights, start_date, end_date):
    prices_df = yf.download(tickers, start=start_date, end=end_date, adjusted=True)
    
    returns_df = prices_df['Close'].pct_change().dropna()
    
    avg_returns = returns_df.mean() * 365
    cov_mat = returns_df.cov() * 365
    
    # RETURNS / EXPECTED RETURNS
    weights = np.array(weights)
    portf_rtns = np.dot(weights, avg_returns)

    # VOLATILITY / STANDARD DEVIATION
    portf_vol = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights)))

    portf_vol = np.array(portf_vol)

    # SHARPE RATIO
    portf_sharpe_ratio = portf_rtns / portf_vol
    
    return portf_rtns.round(2), portf_vol.round(2), portf_sharpe_ratio.round(2)

# this section controls callbacks for recommended portfolio
@app.callback(
    [Output("vol_rtn_input", "min"),
    Output("vol_rtn_input", "max"),
    Output("vol_rtn_input", "value"),
    Output("portfolio_tables", "data"),
    Output("portfolio_metrics", "children"),
    Output("vol_rtn_input", "disabled"),
    Output("vol_rtn_label", "children"),
    Output("recommended_alert_auto", 'children'),
    Output("recommended_alert_auto", 'color'),
    Output("recommended_alert_auto", 'is_open'),],
    [Input("tabs", "active_tab"),
    Input('submit_vol_rtn_input', 'n_clicks'),
    Input("mpt_dataframe", "data"),
    Input("top_10_tickers", "data")],
    [State('vol_rtn_input', 'value'),
    State('recommended_alert_auto', 'is_open'),]
)
def switch_tab(at, n_clicks, data, top_10_crypto, slider_val, is_open):
    n_portfolios = 10 ** 5
    n_days = 365
    portf_results_df = pd.DataFrame(data)
    # print("portf_results_df : " + str(portf_results_df))
    risky_assets = top_10_crypto
    n_assets = len(risky_assets)

    # Simulate random portfolio weights
    np.random.seed(42)
    weights = np.random.random(size=(n_portfolios, n_assets))
    weights /=  np.sum(weights, axis=1)[:, np.newaxis]

    is_open = True

    # for best sharpe ratio
    if at == "best":
        max_sharpe_ind, max_sharpe_portf_weights = max_sharpe(portf_results_df, weights)

        table_data = []
        for i in range(n_assets):
            pair = {'crypto': risky_assets[i], 'weightage': max_sharpe_portf_weights[i].round(2)}
            # print(pair)
            table_data.append(pair)

        metrics = portf_results_df.loc[max_sharpe_ind]
        # recommended_df = pd.DataFrame(table_data, columns=['crypto', 'weights'])
        recommended_metrics = pd.DataFrame({"Sharpe Ratio": [metrics[1].round(2)], "Returns": [metrics[0].round(2)], "Volatility": [metrics[2].round(2)]})
        
        fig  = dbc.Table.from_dataframe(recommended_metrics, striped=True, bordered=True, hover=True)
        # fig = dbc.Table.from_dataframe(recommended_df, striped=True, bordered=True, hover=True)
        
        min_vol_value = 0
        max_vol_value = 0
        min_vol_value = 0
        label_text = "Input disabled"
        text = ""
        color = ""
        is_open = False
        return min_vol_value, max_vol_value, min_vol_value, table_data, fig, True, label_text, text, color, is_open

    # adjusting volatility
    elif at == "volatility":
        min_vol_value, max_vol_value = get_min_max_vol(portf_results_df)

        if port_exists_vol_input(slider_val, n_portfolios, portf_results_df):
            index_from_vol = portf_ind_from_vol(slider_val, n_portfolios, portf_results_df)
            weights_from_vol = weights[index_from_vol]
            
            # display tables
            table_data = []
            for i in range(n_assets):
                pair = {'crypto': risky_assets[i], 'weightage': weights_from_vol[i].round(2)}
                # print(pair)
                table_data.append(pair)
            
            metrics = portf_results_df.loc[index_from_vol]
            # recommended_df = pd.DataFrame(table_data, columns=['crypto', 'weights'])
            recommended_metrics = pd.DataFrame({"Sharpe Ratio": [metrics[1].round(2)], "Returns": [metrics[0].round(2)], "Volatility": [metrics[2].round(2)]})

            fig = dbc.Table.from_dataframe(recommended_metrics, striped=True, bordered=True, hover=True)
            # fig2 = dbc.Table.from_dataframe(recommended_df, striped=True, bordered=True, hover=True)
            
            label_text = "Input volatility"
            text = "Success!"
            color = "success"

            return min_vol_value, max_vol_value, slider_val, table_data, fig, False, label_text, text, color, is_open

        else:
            index_from_vol = portf_ind_from_vol(min_vol_value, n_portfolios, portf_results_df)
            weights_from_vol = weights[index_from_vol]
            
            # display tables
            table_data = []
            for i in range(n_assets):
                pair = {'crypto': risky_assets[i], 'weightage': weights_from_vol[i].round(2)}
                # print(pair)
                table_data.append(pair)

            metrics = portf_results_df.loc[index_from_vol]
            # recommended_df = pd.DataFrame(table_data, columns=['crypto', 'weights'])
            recommended_metrics = pd.DataFrame({"Sharpe Ratio": [metrics[1].round(2)], "Returns": [metrics[0].round(2)], "Volatility": [metrics[2].round(2)]})

            fig = dbc.Table.from_dataframe(recommended_metrics, striped=True, bordered=True, hover=True)
            # fig2 = dbc.Table.from_dataframe(recommended_df, striped=True, bordered=True, hover=True)
            if slider_val == 0:
                label_text = ""
                text = ""
                color = ""
                is_open = False

                return min_vol_value, max_vol_value, min_vol_value, table_data, fig, False, label_text, text, color, is_open

            else:
                label_text = "Input volatility"
                text = "No portfolio for specified volatility"
                color = "danger"

                return min_vol_value, max_vol_value, slider_val, table_data, fig, False, label_text, text, color, is_open

    # adjusting returns
    elif at == "returns":
        min_vol_value, max_vol_value = get_min_max_vol(portf_results_df)
        min_vol_ind = portf_ind_from_vol(min_vol_value, n_portfolios, portf_results_df)
        min_rtn_value, max_rtn_value = get_min_max_rtns(portf_results_df, min_vol_ind)

        if port_exists_rtn_input(slider_val, n_portfolios, portf_results_df):
            index_from_rtn = portf_ind_from_rtn(slider_val, n_portfolios, portf_results_df)
            weights_from_rtn = weights[index_from_rtn]
            
            # display tables
            table_data = []
            for i in range(n_assets):
                pair = {'crypto': risky_assets[i], 'weightage': weights_from_rtn[i].round(2)}
                # print(pair)
                table_data.append(pair)

            metrics = portf_results_df.loc[index_from_rtn]
            # recommended_df = pd.DataFrame(table_data, columns=['crypto', 'weights'])
            recommended_metrics = pd.DataFrame({"Sharpe Ratio": [metrics[1].round(2)], "Returns": [metrics[0].round(2)], "Volatility": [metrics[2].round(2)]})
            fig = dbc.Table.from_dataframe(recommended_metrics, striped=True, bordered=True, hover=True)
            # fig2 = dbc.Table.from_dataframe(recommended_df, striped=True, bordered=True, hover=True)
            
            label_text = "Input returns"
            text = "Success!"
            color = "success"

            return min_rtn_value, max_rtn_value, slider_val, table_data, fig, False, label_text, text, color, is_open

        else: 
            index_from_rtn = portf_ind_from_rtn(max_rtn_value, n_portfolios, portf_results_df)
            weights_from_rtn = weights[index_from_rtn]
            
            # display tables
            table_data = []
            for i in range(n_assets):
                pair = {'crypto': risky_assets[i], 'weightage': weights_from_rtn[i].round(2)}
                # print(pair)
                table_data.append(pair)

            metrics = portf_results_df.loc[index_from_rtn]
            # recommended_df = pd.DataFrame(table_data, columns=['crypto', 'weights'])
            recommended_metrics = pd.DataFrame({"Sharpe Ratio": [metrics[1].round(2)], "Returns": [metrics[0].round(2)], "Volatility": [metrics[2].round(2)]})

            fig = dbc.Table.from_dataframe(recommended_metrics, striped=True, bordered=True, hover=True)
            # fig2 = dbc.Table.from_dataframe(recommended_df, striped=True, bordered=True, hover=True)
            if slider_val == 0:
                label_text = ""
                text = ""
                color = ""
                is_open = False

                return min_rtn_value, max_rtn_value, max_rtn_value, table_data, fig, False, label_text, text, color, is_open
                
            else:
                text = "No portfolio for specified returns"
                color = "danger"
                label_text = "Input returns"

                return min_rtn_value, max_rtn_value, slider_val, table_data, fig, False, label_text, text, color, is_open

def max_sharpe(portf_results_df, weights):
    max_sharpe_ind = np.argmax(portf_results_df.sharpe_ratio)
    max_sharpe_portf_weights = weights[max_sharpe_ind]
    
    return max_sharpe_ind, max_sharpe_portf_weights

def get_min_max_vol(portf_results_df):
    min_vol_ind = np.argmin(portf_results_df.volatility)
    min_vol_value = portf_results_df['volatility'][min_vol_ind].round(2)

    max_vol_ind = np.argmax(portf_results_df.volatility)
    max_vol_value = portf_results_df['volatility'][max_vol_ind].round(2)
    return min_vol_value, max_vol_value

def port_exists_vol_input(vol_input, n_portfolios, portf_results_df):
    for i in range(n_portfolios):
        if portf_results_df['volatility'][i].round(2) == vol_input:
            return True
    return False

def portf_ind_from_vol(vol_input, n_portfolios, portf_results_df):
    portfolios_consider = {}
    for i in range(n_portfolios):
        if portf_results_df['volatility'][i].round(2) == vol_input:
            portfolios_consider[i] = portf_results_df['returns'][i]
            found = True


    max_key = max(portfolios_consider, key=portfolios_consider.get)
    return max_key

def get_min_max_rtns(portf_results_df, min_vol_ind):
    min_rtn_value = portf_results_df['returns'][min_vol_ind].round(2)

    max_rtn_ind = np.argmax(portf_results_df.returns)
    max_rtn_value = portf_results_df['returns'][max_rtn_ind].round(2)
    return min_rtn_value,max_rtn_value

def port_exists_rtn_input(rtn_input, n_portfolios, portf_results_df):
    for i in range(n_portfolios):
        if portf_results_df['returns'][i].round(2) == rtn_input:
            return True
    return False

def portf_ind_from_rtn(rtn_input, n_portfolios, portf_results_df):
    portfolios_consider = {}
    for i in range(n_portfolios):
        if portf_results_df['returns'][i].round(2) == rtn_input:
            portfolios_consider[i] = portf_results_df['volatility'][i]
    
    min_key = min(portfolios_consider, key=portfolios_consider.get)

    return min_key

# this section controls callbacks for portfolio comparisons
@app.callback(
    [Output('recommended_weights', 'children'),
    Output('gut_feel_weights', 'children'),
    Output('recommended_returns', 'children'),
    Output('gut_feel_returns', 'children'),
    Output('recommended_vol', 'children'),
    Output('gut_feel_vol', 'children'),
    Output('portf_comparison_alert', 'is_open'),
    Output('portf_comparison_alert', 'color'),
    Output('portf_comparison_alert', 'children')],
    Input('submit_comparison', 'n_clicks'),
    [State('portfolio_table', 'data'),
    State('portfolio_tables', 'data'),
    State('start_analyze_date', 'value'),
    State('end_analyze_date', 'value')],
    prevent_initial_call = True
)
def update_portfolio_comparison(clicks, gut_feel_data, reco_data, start_date, end_date):
    # print(reco_data)
    gut_feel_tickers = []
    gut_feel_weights = []
    for k in range(len(gut_feel_data)):
        gut_feel_tickers.append(gut_feel_data[k]['crypto'])
        gut_feel_weights.append(gut_feel_data[k]['weightage'])
    
    reco_tickers = []
    reco_weights = []
    for i in range(len(reco_data)):
        reco_tickers.append(reco_data[i]['crypto'])
        reco_weights.append(reco_data[i]['weightage'])
    
    print(round(sum(gut_feel_weights), 1) == 1.0)
    print(round(sum(gut_feel_weights), 1) == 1)
    print(type(gut_feel_weights[0]))
    if isinstance(gut_feel_weights[0], str) or round(sum(gut_feel_weights), 1) != 1.0:
        text = "Your total weights for you selected portfolio is not 1!"
        color = "danger"
        is_open = True
        empty_div = html.Div()
        return empty_div, empty_div, empty_div, empty_div, empty_div, empty_div, is_open, color, text
    else:
        recommended_pie = dcc.Graph(figure=px.pie(reco_data, names='crypto', values='weightage'))
        gut_feel_pie = dcc.Graph(figure=px.pie(gut_feel_data, names='crypto', values='weightage'))

        recommended_returns_series = get_rtn_series_df(reco_tickers, reco_weights, start_date, end_date)
        recommended_returns_df = recommended_returns_series.to_frame().reset_index()
        recommended_returns = dcc.Graph(
                                    figure = {
                                        'data': [
                                            {'x': recommended_returns_df['Date'], 'y': recommended_returns_df[0],'type':'line'}
                                        ],
                                        'layout' : {
                                            'title': 'Recommended Portfolio Returns'
                                        }
                                    }
                                )

        gut_feel_returns_series = get_rtn_series_df(gut_feel_tickers, gut_feel_weights, start_date, end_date)
        gut_feel_returns_series = gut_feel_returns_series.rename("Close")
        # print(gut_feel_returns_series)
        gut_feel_returns_df = gut_feel_returns_series.to_frame().reset_index()
        
        gut_feel_returns = dcc.Graph(
            figure = {
                'data': [
                    {'x': gut_feel_returns_df['Date'], 'y': gut_feel_returns_df['Close'],'type':'line'}
                ],
                'layout' : {
                    'title': 'Selected Portfolio Returns'
                }
            }
        )

        recommended_vol_df = get_rolling_volatility_df(recommended_returns_series, 20).to_frame().reset_index()
        print(recommended_vol_df)
        recommended_vol = dcc.Graph(
                                figure = {
                                    'data': [
                                        {'x': recommended_vol_df['Date'], 'y': recommended_vol_df[0],'type':'line'}
                                    ],
                                    'layout' : {
                                        'title': 'Recommended Portfolio Volatility'
                                    }
                                }
                            )

        gut_feel_vol_df = get_rolling_volatility_df(gut_feel_returns_series, 20).to_frame().reset_index()
        # print(gut_feel_vol_df)
        gut_feel_vol = dcc.Graph(
                            figure = {
                                'data': [
                                    {'x': gut_feel_vol_df['Date'], 'y': gut_feel_vol_df['Close'],'type':'line'}
                                ],
                                'layout' : {
                                    'title': 'Selected Portfolio Volatility'
                                }
                            }
                        )
        
        text = ""
        color = ""
        is_open = False

        return recommended_pie, gut_feel_pie, recommended_returns, gut_feel_returns, recommended_vol, gut_feel_vol, is_open, color, text

def get_rtn_series_df(tickers, weights, start_date, end_date):
    prices_df = yf.download(tickers, start=start_date, end=end_date, adjusted=True)
    close_df = prices_df['Close']
    rtn_series = (close_df.pct_change()+1).cumprod() - 1 
    # print(len(weights))
    # print(len(rtn_series))
    weighted_rtn_series = weights * rtn_series
    if isinstance(weighted_rtn_series, pd.Series):
        final_rtn_series = weighted_rtn_series
        final_rtn_series.rename("Close")
    else:
        final_rtn_series = weighted_rtn_series.sum(axis=1)
        final_rtn_series.rename("Close")
    
    return final_rtn_series

def get_rolling_volatility_df(rtn_series, rolling_window):
    return pd.DataFrame.rolling(np.log((rtn_series+1)/(rtn_series+1).shift(1)),window=rolling_window).std() * np.sqrt(365)

@app.callback(
    [Output("lstm_table", "children"),
    Output("lstm_graph", "children")],
    Input("lstm_button", "n_clicks"),
    [State("portfolio_tables", "data"),
    State('start_analyze_date', 'value'),
    State('end_analyze_date', 'value')],
    prevent_initial_call = True
)
def lstm_predictions(clicks, data, start_date, end_date):
    # get crypto and weights list
    crypto_list = []
    weights_list = []
    for item in data:
        crypto_list.append(item["crypto"])
        weights_list.append(item["weightage"])

    prices_df = yf.download(crypto_list, start=start_date, end=end_date, adjusted=True)
    prices_df = prices_df['Close']
    results = []

    original_price = []
    for ticker in crypto_list:
        current_price = prices_df[ticker].iloc[-1]
        original_price.append(current_price)
    original_price
    dataFrames_arr = []

    for ticker in crypto_list:

        close_df = prices_df[ticker]


        scaler=MinMaxScaler(feature_range=(0,1))
        close_df=scaler.fit_transform(np.array(close_df).reshape(-1,1))

        training_size=int(len(close_df)*0.65)
        # test_size=len(close_df)-training_size
        train_data,test_data=close_df[0:training_size,:],close_df[training_size:len(close_df),:1]


        time_step = 100
        X_train, y_train = create_dataset(train_data, time_step)
        X_test, ytest = create_dataset(test_data, time_step)

        X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
        X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

        model= Sequential()
        model.add(LSTM(50,return_sequences=True,input_shape=(X_train.shape[1],X_train.shape[2])))
        model.add(LSTM(50,return_sequences=True))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error',optimizer='adam') 
        model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=1,batch_size = 32,verbose =1)

        shape_input = test_data.shape[0]-100


        x_input=test_data[shape_input:].reshape(1,-1)

        temp_input=list(x_input)
        temp_input=temp_input[0].tolist()



        lst_output=[]
        n_steps=100
        i=0
        while(i<30):
            
            if(len(temp_input)>100):
                #print(temp_input)
                x_input=np.array(temp_input[1:])
                # print("{} day input {}".format(i,x_input))
                x_input=x_input.reshape(1,-1)
                x_input = x_input.reshape((1, n_steps, 1))
                #print(x_input)
                yhat = model.predict(x_input, verbose=0)
                # print("{} day output {}".format(i,yhat))
                temp_input.extend(yhat[0].tolist())
                temp_input=temp_input[1:]
                #print(temp_input)
                lst_output.extend(yhat.tolist())
                i=i+1
            else:
                x_input = x_input.reshape((1, n_steps,1))
                yhat = model.predict(x_input, verbose=0)
                # print(yhat[0])
                temp_input.extend(yhat[0].tolist())
                # print(len(temp_input))
                lst_output.extend(yhat.tolist())
                i=i+1


        df3=close_df.tolist()
        df3.extend(lst_output)

        df3=scaler.inverse_transform(df3).tolist()
        dataFrames_arr.append(df3)

        results.append(df3[-1])

    future_prices = []

    for pr in results:
        future_prices.append(pr[0])

    array = []

    for i in range(len(original_price)):
        eachTicker = []
        eachTicker.append(round(original_price[i], 4))
        eachTicker.append(round(future_prices[i], 4))
        array.append(eachTicker)


    time_extension = len(df3) 

    EndDate = prices_df.index[0] + timedelta(days=time_extension)

    dateIndexes = pd.date_range(start='2020-01-01', end=EndDate)

    df = pd.DataFrame(data = array, 
                      index = crypto_list, 
                      columns = ['Current Price','Predicted Price'])
    df['Predicted % Change'] = round(((df['Predicted Price'] - df['Current Price'])/df['Current Price']) * 100, 4)
    df['Weights from MPT'] = weights_list
    df.sort_values(by='Predicted % Change',ascending = False,inplace = True)
    df.index.names = ['Crypto']

    # pin2.1
    new_df = df.copy()
    # new_df['Weights'] = weights_list
    new_weights = new_df['Weights from MPT']

    # LSTM weights stuff
    # weights_list_series has to be sorted according to predicted % change
    weights_list_series = numpy.array([float(weight) for weight in new_weights])
    mpt_weightage = numpy.array([[0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9]])
    lstm_weights = numpy.array([0.24, 0.21, 0.15, 0.13, 0.09, 0.07, 0.05, 0.03, 0.02, 0.01])
    lstm_weightage = numpy.array([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])
    final_weights = weights_list_series * mpt_weightage + lstm_weights * lstm_weightage
    new_df['Weights Considering Predicted Prices'] = final_weights[0].round(2)

    name = crypto_list[0]
    column_name = name + " Price"
    each_df = pd.DataFrame(data = dataFrames_arr[0], 
                      index = dateIndexes, 
                      columns = [column_name])
      


    for j in range(1, len(dataFrames_arr)):
        name = crypto_list[j]
        column_name = name + " Price"
        each_df2 = pd.DataFrame(data = dataFrames_arr[j], 
                        index = dateIndexes, 
                        columns = [column_name])
        each_df = pd.merge(each_df, each_df2, left_index=True, right_index=True)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for k in range(len(crypto_list)):
        name = crypto_list[k]
        column_name = name + " Price"

        # Add traces
        fig.add_trace(
            go.Scatter(x=dateIndexes, y=each_df[column_name], name=column_name),
            secondary_y=False,
        )

    # fig.show()
    # new_df

    figure = dcc.Graph(figure = fig)
    # print(new_df)
    new_df = new_df.reset_index()
    table = dbc.Table.from_dataframe(new_df, striped = True , bordered = True, hover = True)

    return table, figure

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
        
    return np.array(dataX), np.array(dataY)

if __name__ == "__main__":
    app.run_server(debug=True)