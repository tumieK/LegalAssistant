import requests
import time

USER_AGENT = "streamlit-legal-assistant/1.0 (youremail@example.com)"

def geocode_address(address):
    """Try to find (lat, lon) for a full SA address with fallbacks."""
    if not address or not address.strip():
        return None, None

    url = "https://nominatim.openstreetmap.org/search"
    headers = {"User-Agent": USER_AGENT}
    params = {"q": address, "format": "json", "limit": 1}

    try:
        # Try the full address first
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        data = resp.json()
        if data:
            lat, lon = float(data[0]["lat"]), float(data[0]["lon"])
            return lat, lon

        # 🪄 Fallback 1: remove house number
        parts = address.split(",")
        if len(parts) > 1:
            short_address = ", ".join(parts[1:])
            params["q"] = short_address
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            data = resp.json()
            if data:
                lat, lon = float(data[0]["lat"]), float(data[0]["lon"])
                return lat, lon

        # 🪄 Fallback 2: city + province + country
        if "Bloemfontein" in address:
            fallback = "Bloemfontein, Free State, South Africa"
            params["q"] = fallback
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            data = resp.json()
            if data:
                lat, lon = float(data[0]["lat"]), float(data[0]["lon"])
                return lat, lon

        return None, None
    except Exception as e:
        print("❌ Geocode failed:", e)
        return None, None
