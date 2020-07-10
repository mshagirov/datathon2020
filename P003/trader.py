
def trader_simulation(trading_hours,Y_pred,Y_targ,
                     INITIAL_CASH = 10000000, SALE_PRICE = 10 , SPOT_PRICE = 20, PENALTY_PRICE = 100,
                     print_at_end=False, debug=False):
    '''
    Simulate trading for "trading_hours" times (freq='h'), Y_pred : predictions (kWh), Y_targ : actual power produced (kWh)

    Returns PROFITS (scalar value)
    '''
    CASH_AT_HAND = INITIAL_CASH # my trading wallet
    cash_history = [CASH_AT_HAND]

    for k,t in enumerate(trading_hours):
        # Sale
        yk, pk = (Y_targ[k], Y_pred[k]) # actual and prediction
        CASH_AT_HAND += max([0.0, min([yk, pk])])*SALE_PRICE # sale amount; min amount to sell is 0
        wasted_cents = (yk-pk)*SALE_PRICE if yk>pk else 0

        # Buy from spot market and Penalty
        available = max([0.0,CASH_AT_HAND/SPOT_PRICE]) # available kWh from spot
        shortfall = pk-yk if pk>yk else 0.0 # shortfall energy
        # buy
        CASH_AT_HAND -= min([available,shortfall])*SPOT_PRICE
        # penalty
        penalty = PENALTY_PRICE*(shortfall-available) if available<=shortfall else 0
        CASH_AT_HAND -= penalty

        # record history
        cash_history.append(CASH_AT_HAND)
        if debug:
            print(f"{str(t)}-> Act.Prod/Pred.:{yk:.1f} /{pk:.1f} (wasted:{wasted_cents:.0f}c); "+
            f"Short./Avail.:{shortfall:.1f} /{available:.1f}; "+
            f"Penalty:{penalty}; Cash:{CASH_AT_HAND:.0f}")
    PROFIT = CASH_AT_HAND-INITIAL_CASH
    if print_at_end:
        print(f'Net profit(euro cents):{PROFIT}')
    return PROFIT
