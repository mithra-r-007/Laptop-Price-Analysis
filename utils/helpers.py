"""
Helper utilities for Laptop Price Analyzer & Prediction System.
Handles currency formatting, data validation, and common utilities.
"""

import re
import pandas as pd


def format_currency_inr(amount):
    """
    Format a number into Indian Rupee (INR) currency format.
    Example: 72450 -> '₹72,450', 128990 -> '₹1,28,990'
    """
    try:
        val = float(amount)
        if pd.isna(val):
            return "₹0"
        
        # Round to integer for clean display
        val_int = int(round(val))
        is_negative = val_int < 0
        s = str(abs(val_int))
        
        if len(s) <= 3:
            formatted = s
        else:
            # Indian numbering: last 3 digits, then groups of 2 digits
            last_three = s[-3:]
            remaining = s[:-3]
            groups = []
            while len(remaining) > 2:
                groups.insert(0, remaining[-2:])
                remaining = remaining[:-2]
            if remaining:
                groups.insert(0, remaining)
            formatted = ",".join(groups) + "," + last_three

        prefix = "-₹" if is_negative else "₹"
        return f"{prefix}{formatted}"
    except Exception:
        return f"₹{amount}"


def clean_numeric_text(val):
    """Extract first numeric float from a string (e.g. '16GB' -> 16.0, '1.65 kg' -> 1.65)"""
    if pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    match = re.search(r"[-+]?\d*\.?\d+", str(val).replace(",", ""))
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


def validate_prediction_inputs(inputs: dict):
    """
    Validates user prediction inputs against basic sanity rules.
    Returns (is_valid, list_of_error_messages)
    """
    errors = []
    if inputs.get("RAM", 0) <= 0:
        errors.append("RAM must be greater than 0 GB.")
    if inputs.get("Storage", 0) <= 0:
        errors.append("Storage must be greater than 0 GB.")
    if inputs.get("Screen_Size", 0) <= 0:
        errors.append("Screen Size must be greater than 0 inches.")
    if inputs.get("Weight_kg", 0) <= 0:
        errors.append("Weight must be greater than 0 kg.")
    return len(errors) == 0, errors
