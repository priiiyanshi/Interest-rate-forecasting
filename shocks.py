def apply_rate_shocks(series, shock_type):
    if shock_type == "+50bps":
        return series + 0.50
    elif shock_type == "+100bps":
        return series + 1.00
    elif shock_type == "-50bps":
        return series - 0.50
    else:
        return series
