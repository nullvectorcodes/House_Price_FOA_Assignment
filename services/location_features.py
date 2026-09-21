"""
Location Features Integration Module.
Converts human-readable location queries into validated geographic coordinates,
Haversine city-center distance, and nearby POI features for the Machine Learning pipeline.
"""

from typing import Any, Dict, Optional
from services.geocoding_service import GeocodingService
from services.poi_service import CITY_CENTROIDS, POIService, calculate_distance


_geocoding_service = GeocodingService()
_poi_service = POIService(default_radius_km=2.0)


def extract_location_features(
    location_query: str,
    radius_km: float = 2.0,
    geocoder: Optional[GeocodingService] = None,
    poi_service: Optional[POIService] = None,
) -> Dict[str, Any]:
    """
    Extracts all location and POI features from a location string.

    Parameters
    ----------
    location_query : str
        Address, locality, postal code, or 'lat, lon'.
    radius_km : float, default=2.0
        Search radius in km for counting amenities.
    geocoder : GeocodingService, optional
    poi_service : POIService, optional

    Returns
    -------
    Dict[str, Any]
        Dictionary containing coordinates, address metadata, distance to city center,
        and nearby POI metrics formatted for ML ingestion and UI display.
    """
    geo = geocoder or _geocoding_service
    poi = poi_service or _poi_service

    geo_res = geo.geocode(location_query)
    if not geo_res:
        raise ValueError(
            f"Unable to geocode location '{location_query}'. Please verify spelling or try another landmark/city."
        )

    lat = float(geo_res["latitude"])
    lon = float(geo_res["longitude"])
    city = str(geo_res.get("city", "Hyderabad"))
    state = str(geo_res.get("state", "Telangana"))
    postal_code = str(geo_res.get("postal_code", "500001"))
    formatted_address = str(geo_res.get("formatted_address", location_query))

    # Calculate Haversine distance to city center
    city_center_coords = CITY_CENTROIDS.get(city.title(), CITY_CENTROIDS["Hyderabad"])
    dist_to_city_center = calculate_distance(lat, lon, city_center_coords[0], city_center_coords[1])
    dist_to_city_center = round(dist_to_city_center, 2)

    # POI metrics within search radius
    schools_count, nearest_school_km = poi.get_category_metrics(lat, lon, "school", radius_km)
    hospitals_count, nearest_hospital_km = poi.get_category_metrics(lat, lon, "hospital", radius_km)
    metro_count, nearest_metro_km = poi.get_category_metrics(lat, lon, "metro", radius_km)
    parks_count, nearest_park_km = poi.get_category_metrics(lat, lon, "park", radius_km)
    shopping_count, nearest_shopping_km = poi.get_category_metrics(lat, lon, "shopping", radius_km)
    supermarket_count, nearest_supermarket_km = poi.get_category_metrics(lat, lon, "supermarket", radius_km)

    return {
        # Raw coordinates & address
        "latitude": round(lat, 5),
        "longitude": round(lon, 5),
        "city": city,
        "state": state,
        "postal_code": postal_code,
        "formatted_address": formatted_address,
        "source": geo_res.get("source", "geocoding"),

        # Geographic distances & POI density metrics (Used in ML)
        "distance_to_city_center": dist_to_city_center,
        "distance_to_city_km": dist_to_city_center,  # Alias for model compatibility
        "schools_within_2km": schools_count,
        "nearest_school_km": nearest_school_km,
        "distance_to_school_km": nearest_school_km,  # Alias for model compatibility
        "hospitals_within_2km": hospitals_count,
        "nearest_hospital_km": nearest_hospital_km,
        "distance_to_hospital_km": nearest_hospital_km,  # Alias for model compatibility
        "metro_within_2km": metro_count,
        "nearest_metro_km": nearest_metro_km,
        "parks_within_2km": parks_count,
        "nearest_park_km": nearest_park_km,
        "shopping_within_2km": shopping_count,
        "nearest_shopping_km": nearest_shopping_km,
        "supermarkets_within_2km": supermarket_count,
        "nearest_supermarket_km": nearest_supermarket_km,
    }
