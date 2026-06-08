from sphero_unsw.toys_scanner import toys_scanner
from sphero_unsw.sphero_edu import SpheroEduAPI

def scan_and_connect():
    """
    Returns (selected_toy, api_context_manager)
    Usage:
        toy, api_cm = scan_and_connect()
        with api_cm as api:
            ...
    """
    scanner = toys_scanner()
    selected_toy = scanner.scan_and_select_toy()
    return selected_toy, SpheroEduAPI(selected_toy)
