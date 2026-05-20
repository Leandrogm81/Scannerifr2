"""Módulo para gerenciamento de risco."""


def calculate_position_size(
    account_value: float, risk_per_trade: float = 0.02, stop_loss_distance: float = 0.1
) -> float:
    """
    Calcula tamanho da posição baseado em risco.

    Args:
        account_value: Valor total da conta
        risk_per_trade: Risco por operação (default 2%)
        stop_loss_distance: Distância do stop loss (default 10%)

    Returns:
        Tamanho da posição em reais
    """
    risk_amount = account_value * risk_per_trade
    position_size = risk_amount / stop_loss_distance
    return min(position_size, account_value)


def calculate_stop_loss(
    entry_price: float, atr: float = None, mult: float = 2.0
) -> float:
    """
    Calcula stop loss.

    Args:
        entry_price: Preço de entrada
        atr: Average True Range (se disponível)
        mult: Múltiplo do ATR ou preço fixo

    Returns:
        Preço de stop loss
    """
    if atr:
        return entry_price - (mult * atr)
    return entry_price * 0.90  # Stop loss fixo de 10% (default)


def calculate_take_profit(
    entry_price: float, atr: float = None, mult: float = 3.0
) -> float:
    """
    Calcula take profit.

    Args:
        entry_price: Preço de entrada
        atr: Average True Range
        mult: Múltiplo do ATR

    Returns:
        Preço de take profit
    """
    if atr:
        return entry_price + (mult * atr)
    return entry_price * 1.15  # Take profit fixo de 15% (default)


def calculate_risk_reward(entry: float, stop: float, target: float) -> float:
    """
    Calcula razão risco/recompensa.

    Args:
        entry: Preço de entrada
        stop: Preço de stop loss
        target: Preço de take profit

    Returns:
        Razão risco/recompensa ( > 1 significa recompensa > risco)
    """
    risk = entry - stop
    reward = target - entry
    return reward / risk if risk > 0 else 0
