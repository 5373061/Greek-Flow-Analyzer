
import os
import sys

print('Working directory:', os.getcwd())

modules_to_check = [
    'api_fetcher', 
    'greek_flow.flow', 
    'entropy_analyzer.entropy_analyzer', 
    'data_loader',
    'plot_helpers',
    'models.ml.regime_classifier',
    'tools.trade_dashboard'
]

for module in modules_to_check:
    try:
        __import__(module)
        print(f'{module}: Available')
    except ImportError as e:
        print(f'{module}: Not found - {str(e)}')
