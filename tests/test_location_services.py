"""
Unit tests for Location and POI Services.
Verifies Haversine calculations, geocoding resolution, POI metrics, and fallback behavior.
"""

import pytest

from services.geocoding_service import GeocodingService
from services.location_features import extract_location_features
from services.poi_service import POIService, calculate_distance


def test_haversine_distance_calculation():
    """Verify mathematical correctness of Haversine distance calculation."""
    # Distance from a point to itself should be 0.0
    assert pytest.approx(calculate_distance(17.3850, 78.4867, 17.3850, 78.4867), abs=1e-5) == 0.0

    # Distance between Hyderabad (17.3850, 78.4867) and Bangalore (12.9716, 77.5946)
    # Known geodesic distance is approximately 500 km (within 490 - 510 km)
    dist = calculate_distance(17.3850, 78.4867, 12.9716, 77.5946)
    assert 490.0 < dist < 510.0


def test_geocoding_resolution():
    """Verify that Geocoding resolves coordinates for major Indian localities."""
    geo = GeocodingService()

    res_hyd = geo.geocode("Gachibowli, Hyderabad")
    assert res_hyd is not None
    assert "latitude" in res_hyd
    assert "longitude" in res_hyd
    assert 17.0 < res_hyd["latitude"] < 18.0
    assert 78.0 < res_hyd["longitude"] < 79.0

    # Test Bangalore
    res_blr = geo.geocode("Koramangala, Bangalore")
    assert res_blr is not None
    assert 12.5 < res_blr["latitude"] < 13.5
    assert 77.0 < res_blr["longitude"] < 78.0


def test_geocoding_coordinates_input():
    """Verify that direct coordinate strings are parsed correctly."""
    geo = GeocodingService()
    res = geo.geocode("17.4401, 78.3489")
    assert res is not None
    assert pytest.approx(res["latitude"], abs=1e-4) == 17.4401
    assert pytest.approx(res["longitude"], abs=1e-4) == 78.3489


def test_poi_metrics_extraction():
    """Verify that POI service retrieves counts and nearest distances."""
    poi = POIService(default_radius_km=2.0)
    # Test near Gachibowli coordinates
    count, nearest_km = poi.get_category_metrics(17.4401, 78.3489, "hospital")
    assert count >= 1
    assert 0.0 < nearest_km <= 5.0

    count_school, nearest_school = poi.get_category_metrics(17.4401, 78.3489, "school")
    assert count_school >= 1
    assert 0.0 < nearest_school <= 5.0


def test_end_to_end_location_features_extraction():
    """Verify extract_location_features constructs full ML dictionary."""
    feats = extract_location_features("Gachibowli, Hyderabad", radius_km=2.0)

    required_keys = [
        "latitude",
        "longitude",
        "city",
        "state",
        "distance_to_city_center",
        "schools_within_2km",
        "nearest_school_km",
        "hospitals_within_2km",
        "nearest_hospital_km",
        "metro_within_2km",
        "nearest_metro_km",
        "parks_within_2km",
        "shopping_within_2km",
        "supermarkets_within_2km",
    ]
    for key in required_keys:
        assert key in feats, f"Missing key '{key}' in extracted location features."
        assert feats[key] is not None
