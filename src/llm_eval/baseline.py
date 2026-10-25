def predict_route(text: str) -> str:
    """Predict a support category using simple keyword rules."""
    text = text.lower()

    if any(word in text for word in ["password", "log in", "login", "account"]):
        return "account_support"

    if any(word in text for word in ["charged", "payment", "billing", "stolen"]):
        return "billing_support"

    if any(word in text for word in ["order", "delivery", "arrive", "shipping"]):
        return "order_support"

    return "general_support"

def predict_escalation(text: str) -> bool:
    """Flag messages that may need urgent human attention."""
    text = text.lower()

    risk_keywords = [
        "without permission",
        "stolen",
        "fraud",
        "unauthorized",
        "hacked",
    ]

    return any(keyword in text for keyword in risk_keywords)
