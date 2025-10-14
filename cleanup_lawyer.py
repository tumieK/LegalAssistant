import json
import time
from geocode import geocode_address  # uses your existing function

# Load the original JSON
with open("data/lawyers.json", "r", encoding="utf-8") as f:
    data = json.load(f)

cleaned = {}

for category, lawyers in data.items():
    cleaned[category] = []
    for lawyer in lawyers:
        # Rename lng → lon if necessary
        if "lng" in lawyer and "lon" not in lawyer:
            lawyer["lon"] = lawyer.pop("lng")

        lat = lawyer.get("lat", 0)
        lon = lawyer.get("lon", 0)

        # Try to re-geocode if missing or 0.0
        if (lat == 0 or lon == 0) and lawyer.get("address"):
            print(f"🗺️ Re-geocoding: {lawyer['name']} — {lawyer['address']}")
            new_lat, new_lon = geocode_address(lawyer["address"])
            if new_lat and new_lon:
                lawyer["lat"], lawyer["lon"] = new_lat, new_lon
                print(f"✅ Updated: {new_lat:.5f}, {new_lon:.5f}")
                time.sleep(1)
            else:
                print(f"⚠️ Could not geocode {lawyer['address']}, removing...")
                continue  # skip this lawyer entirely

        # Only keep valid coordinates
        if lawyer.get("lat") and lawyer.get("lon"):
            cleaned[category].append(lawyer)

# Save to a new file for safety
with open("data/lawyers_clean.json", "w", encoding="utf-8") as f:
    json.dump(cleaned, f, indent=2)

print("✅ Cleaned data saved to data/lawyers_clean.json")
