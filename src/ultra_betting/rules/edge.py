"""Edge calculation — predicted vs market price."""


def calculate_edge(predicted_bfsp: float, live_price: float, side: str) -> float:
    """Calculate the edge as a percentage.

    For BACK bets: edge is positive when live price > predicted BFSP
    (the market thinks the horse is longer than our model says — value).

    For LAY bets: edge is positive when live price < predicted BFSP
    (the market thinks the horse is shorter than our model says — value to lay).

    Returns:
        Edge as a percentage. Positive = favourable.
    """
    if predicted_bfsp <= 0 or live_price <= 0:
        return 0.0

    if side == "BACK":
        # Value to back: live price is higher than what we think fair odds are
        edge = (live_price / predicted_bfsp - 1) * 100
    else:
        # Value to lay: live price is lower than what we think fair odds are
        edge = (predicted_bfsp / live_price - 1) * 100

    return edge
