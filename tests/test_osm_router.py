import pytest

from src.osm_router import route_from_coordinates


@pytest.mark.skip(reason="Requires internet and OSM download; run manually when needed.")
def test_route_from_coordinates_live() -> None:
    result = route_from_coordinates(
        place_name="Bengaluru, Karnataka, India",
        source_lat=12.9716,
        source_lon=77.5946,
        dest_lat=12.9352,
        dest_lon=77.6245,
        congestion_level="Medium",
    )
    assert result.distance_km > 0
    assert result.eta_min > 0
    assert len(result.coordinate_path) > 1
