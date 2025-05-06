def test_import_and_run():
    from greek_flow import GreekEnergyFlow
    import pandas as pd
    # minimal dummy DF with the required columns
    df = pd.DataFrame([{
        "strike": 100, "expiration": pd.Timestamp.today() + pd.Timedelta(days=7),
        "type": "call", "openInterest": 1, "impliedVolatility": 0.5,
        "delta": 0.5, "gamma": 0.01,
    }])
    market = {"currentPrice": 100, "historicalVolatility": 0.2, "riskFreeRate": 0.04}
    GreekEnergyFlow().analyze_greek_profiles(df, market)
