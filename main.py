import yfinance as yf
import pandas as pd
import seaborn as sns
import PySimpleGUI as sg

import matplotlib.pyplot as plt

def plot_stock_data(stock_name, start_date, end_date):
    data = yf.download(stock_name, start=start_date, end=end_date)
    data.to_csv('C:\\Users\\ASUS\\OneDrive\\Desktop\\PYTHON\\Stock\\stock_data.csv')

    data = yf.Ticker(stock_name)
    stock_data = data.history(start=start_date, end=end_date)
    stock_data.to_csv('C:\\Users\\ASUS\\OneDrive\\Desktop\\PYTHON\\Stock\\stock_data.csv')

    # Plotting the stock data using seaborn
    sns.lineplot(x=stock_data.index, y=stock_data['Close'])
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title(f'{stock_name} Stock Ups and Downs')  # Make the title dynamic
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.show()

# Create the GUI window
window = sg.Window("Stock Data").Layout([[sg.Text("Stock Symbol:"), sg.Input(key='-STOCK-')], [sg.Text("Start Date (YYYY-MM-DD):"), sg.Input(key='-START-')], [sg.Text("End Date (YYYY-MM-DD):"), sg.Input(key='-END-')], [sg.Button("Plot")]])

# Start the GUI event loop
while True:
    event, values = window.read()
    if event == sg.WINDOW_CLOSED:
        break
    if event == 'Plot':
        plot_stock_data(values['-STOCK-'], values['-START-'], values['-END-'])
