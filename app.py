"""
Fixed-Scenario Debris Network Generator
===================================================

Streamlit application for:
1) selecting custom network nodes on a map,
2) routing physical links over OpenStreetMap streets,
3) downloading nearby OSM buildings,
4) estimating damage-state probabilities under one fixed PGA scenario,
5) calculating expected building-debris volume,
6) allocating debris to nearby physical road links without double counting,
7) exporting reproducible Excel and image outputs.
"""

from __future__ import annotations

import io
import re
from math import asin, cos, erf, log, radians, sin, sqrt
from typing import Any

import certifi
import contextily as ctx
import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import streamlit as st
from folium.plugins import Draw
from shapely.geometry import LineString, Point
from shapely.ops import linemerge, unary_union
from streamlit_folium import st_folium


# =============================================================================
# 1. FIXED STUDY CONFIGURATION
# =============================================================================

APP_TITLE = "Fixed-Scenario Debris Network Generator"

MAX_POINTS = 50
MAX_CONNECTION_DISTANCE_M = 150.0
BBOX_PADDING_DEG = 0.001
FLOOR_HEIGHT_M = 3.2

# Fixed earthquake scenario: one intensity value for all buildings.
FIXED_IM_TYPE = "PGA"
FIXED_PGA_G = 0.20
SCENARIO_NAME = "Fixed_PGA_0.20g"

# Geometric debris-to-road allocation assumptions.
ROAD_INFLUENCE_BUFFER_M = 20.0
STREET_DEPOSITION_FACTOR = 0.40
MIN_DEBRIS_THRESHOLD_M3 = 20.0  # افزایش آستانه برای حذف حجم‌های ناچیز

FRAGILITY_ROWS = [
    ("MUR_L", 0.08, 0.65, 0.16, 0.65, 0.28, 0.65, 0.45, 0.65),
    ("MUR_M", 0.06, 0.65, 0.13, 0.65, 0.24, 0.65, 0.40, 0.65),
    ("RC_L", 0.12, 0.60, 0.25, 0.60, 0.45, 0.60, 0.75, 0.60),
    ("RC_M", 0.10, 0.60, 0.22, 0.60, 0.40, 0.60, 0.68, 0.60),
    ("RC_H", 0.08, 0.60, 0.18, 0.60, 0.34, 0.60, 0.60, 0.60),
    ("STEEL", 0.14, 0.60, 0.30, 0.60, 0.55, 0.60, 0.90, 0.60),
    ("WOOD_L", 0.18, 0.65, 0.38, 0.65, 0.70, 0.65, 1.10, 0.65),
]

DEBRIS_RATE_ROWS = [
    ("MUR_L", 0.000, 0.005, 0.030, 0.100, 0.220),
    ("MUR_M", 0.000, 0.006, 0.035, 0.120, 0.240),
    ("RC_L", 0.000, 0.004, 0.025, 0.080, 0.160),
    ("RC_M", 0.000, 0.004, 0.020, 0.070, 0.150),
    ("RC_H", 0.000, 0.003, 0.018, 0.060, 0.140),
    ("STEEL", 0.000, 0.002, 0.015, 0.050, 0.100),
    ("WOOD_L", 0.000, 0.003, 0.020, 0.070, 0.140),
]

FRAGILITY_COLUMNS = [
    "taxonomy",
    "DS1_median_g", "DS1_beta",
    "DS2_median_g", "DS2_beta",
    "DS3_median_g", "DS3_beta",
    "DS4_median_g", "DS4_beta",
]
DEBRIS_COLUMNS = ["taxonomy", "DS0", "DS1", "DS2", "DS3", "DS4"]
DAMAGE_STATES = ["DS0", "DS1", "DS2", "DS3", "DS4"]


# =============================================================================
# 2. OSMNX AND STREAMLIT SETTINGS
# =============================================================================

ox.settings.requests_kwargs = {"verify": certifi.where()}
if hasattr(ox.settings, "requests_timeout"):
    ox.settings.requests_timeout = 180
elif hasattr(ox.settings, "timeout"):
    ox.settings.timeout = 180
if hasattr(ox.settings, "overpass_rate_limit"):
    ox.settings.overpass_rate_limit = True
ox.settings.use_cache = True
ox.settings.cache_folder = "./osm_cache"

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title("🗺️ Fixed-Scenario Debris Network Generator")
st.caption(
    "تولید شبکه خیابانی و حجم مورد انتظار آوار با شدت ثابت زلزله"
)


# =============================================================================
# 3. SMALL UTILITY FUNCTIONS
# =============================================================================


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_radius_m = 6_371_000.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2.0) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2.0) ** 2
    return 2.0 * earth_radius_m * asin(sqrt(a))


def parse_first_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        result = float(value)
        return result if np.isfinite(result) else None

    match = re.search(r"[-+]?\d*\.?\d+", str(value).replace(",", "."))
    if not match:
        return None
    try:
        result = float(match.group())
        return result if np.isfinite(result) else None
    except ValueError:
        return None


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set, np.ndarray)):
        value = ";".join(str(item) for item in value)
    return str(value).strip().lower()


def union_all_geometries(geometries: gpd.GeoSeries):
    try:
        return geometries.union_all()
    except AttributeError:
        return unary_union(geometries.tolist())


def ensure_session_state() -> None:
    defaults = {
        "excel_data": b"",
        "graph_img_data": b"",
        "sat_img_data": b"",
        "stats": "",
        "drawn_points": [],
        "map_layers": None,
        "map_reset_counter": 0,
        "custom_edges_str": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def parse_custom_edges(text: str, number_of_nodes: int) -> list[tuple[int, int]]:
    parsed: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    for raw_pair in text.split(","):
        pair = raw_pair.strip()
        if "-" not in pair:
            continue
        parts = [part.strip() for part in pair.split("-")]
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            continue

        u, v = int(parts[0]), int(parts[1])
        if u == v or u < 0 or v < 0 or u >= number_of_nodes or v >= number_of_nodes:
            continue

        physical_pair = (min(u, v), max(u, v))
        if physical_pair not in seen:
            parsed.append(physical_pair)
            seen.add(physical_pair)

    return parsed


def add_numbered_marker(
    folium_map: folium.Map,
    lat: float,
    lon: float,
    number: int,
    color_bg: str = "rgba(255,255,0,0.9)",
    border_color: str = "red",
    font_size: str = "13pt",
    size: int = 28,
) -> None:
    html = f"""
    <div style="width:{size}px;height:{size}px;line-height:{size}px;text-align:center;
    font-size:{font_size};font-weight:bold;color:black;background-color:{color_bg};
    border-radius:50%;border:2px solid {border_color};
    box-shadow:1px 1px 3px rgba(0,0,0,0.5);">{number}</div>
    """
    folium.Marker(
        location=[lat, lon],
        icon=folium.DivIcon(
            icon_size=(size, size),
            icon_anchor=(size // 2, size // 2),
            html=html,
        ),
        draggable=False,
    ).add_to(folium_map)


# =============================================================================
# 4. PARAMETER TABLES AND DAMAGE/DEBRIS CALCULATION
# =============================================================================


def validate_fragility_table(df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in FRAGILITY_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Fragility CSV missing columns: {missing}")

    result = df[FRAGILITY_COLUMNS].copy()
    result["taxonomy"] = result["taxonomy"].astype(str).str.strip().str.upper()
    numeric_columns = [column for column in FRAGILITY_COLUMNS if column != "taxonomy"]
    result[numeric_columns] = result[numeric_columns].apply(pd.to_numeric, errors="coerce")

    if result[numeric_columns].isna().any().any():
        raise ValueError("Fragility CSV contains invalid numeric values.")
    if (result[[column for column in numeric_columns if "median" in column]] <= 0).any().any():
        raise ValueError("Fragility medians must be positive.")
    if (result[[column for column in numeric_columns if "beta" in column]] <= 0).any().any():
        raise ValueError("Fragility beta values must be positive.")

    return result.drop_duplicates("taxonomy", keep="last").reset_index(drop=True)


def validate_debris_table(df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in DEBRIS_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Debris-rate CSV missing columns: {missing}")

    result = df[DEBRIS_COLUMNS].copy()
    result["taxonomy"] = result["taxonomy"].astype(str).str.strip().str.upper()
    result[DAMAGE_STATES] = result[DAMAGE_STATES].apply(pd.to_numeric, errors="coerce")

    if result[DAMAGE_STATES].isna().any().any():
        raise ValueError("Debris-rate CSV contains invalid numeric values.")
    if (result[DAMAGE_STATES] < 0).any().any():
        raise ValueError("Debris rates cannot be negative.")

    return result.drop_duplicates("taxonomy", keep="last").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_parameter_tables() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    fragility_df = validate_fragility_table(
        pd.DataFrame(FRAGILITY_ROWS, columns=FRAGILITY_COLUMNS)
    )
    debris_df = validate_debris_table(
        pd.DataFrame(DEBRIS_RATE_ROWS, columns=DEBRIS_COLUMNS)
    )

    common_taxonomies = set(fragility_df["taxonomy"]) & set(debris_df["taxonomy"])
    if not common_taxonomies:
        raise ValueError("جدول شکنندگی و نرخ آوار taxonomy مشترک ندارند.")

    fragility_df = fragility_df[
        fragility_df["taxonomy"].isin(common_taxonomies)
    ].copy()
    debris_df = debris_df[
        debris_df["taxonomy"].isin(common_taxonomies)
    ].copy()
    return fragility_df, debris_df, "embedded fixed-scenario parameters"


def infer_levels(row: pd.Series) -> tuple[float, str]:
    levels_value = parse_first_float(row.get("building:levels"))
    if levels_value is not None and levels_value > 0:
        return float(np.clip(round(levels_value), 1, 40)), "building:levels"

    height_value = parse_first_float(row.get("height"))
    if height_value is not None and height_value > 0:
        estimated = max(1, round(height_value / FLOOR_HEIGHT_M))
        return float(np.clip(estimated, 1, 40)), "height"

    building_type = safe_text(row.get("building"))
    defaults = {
        "house": 2,
        "detached": 2,
        "semidetached_house": 2,
        "terrace": 3,
        "apartments": 5,
        "residential": 4,
        "commercial": 3,
        "retail": 2,
        "office": 4,
        "industrial": 1,
        "warehouse": 1,
        "school": 3,
        "hospital": 4,
    }
    return float(defaults.get(building_type, 3)), "default_by_use"


def infer_taxonomy(row: pd.Series, levels: float) -> tuple[str, str]:
    material = safe_text(row.get("building:material"))
    building_type = safe_text(row.get("building"))
    structure = safe_text(row.get("building:structure"))
    combined = " ".join([material, building_type, structure])

    if any(keyword in combined for keyword in ["wood", "timber", "wood_frame"]):
        return "WOOD_L", "material"

    if any(keyword in combined for keyword in ["steel", "metal"]):
        return "STEEL", "material"

    if any(keyword in combined for keyword in ["concrete", "reinforced_concrete", "cement"]):
        if levels <= 3:
            return "RC_L", "material+height"
        if levels <= 7:
            return "RC_M", "material+height"
        return "RC_H", "material+height"

    if any(
        keyword in combined
        for keyword in ["brick", "masonry", "stone", "sandstone", "limestone"]
    ):
        return ("MUR_L", "material+height") if levels <= 3 else ("MUR_M", "material+height")

    if building_type in {"industrial", "warehouse", "hangar"}:
        return "STEEL", "use_assumption"

    if levels <= 3:
        return "MUR_L", "height_assumption"
    if levels <= 5:
        return "MUR_M", "height_assumption"
    if levels <= 7:
        return "RC_M", "height_assumption"
    return "RC_H", "height_assumption"


def fragility_lookup_from_df(df: pd.DataFrame) -> dict[str, dict[str, tuple[float, float]]]:
    lookup: dict[str, dict[str, tuple[float, float]]] = {}
    for _, row in df.iterrows():
        taxonomy = str(row["taxonomy"])
        lookup[taxonomy] = {
            state: (
                float(row[f"{state}_median_g"]),
                float(row[f"{state}_beta"]),
            )
            for state in ["DS1", "DS2", "DS3", "DS4"]
        }
    return lookup


def debris_lookup_from_df(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        str(row["taxonomy"]): {state: float(row[state]) for state in DAMAGE_STATES}
        for _, row in df.iterrows()
    }


def damage_state_probabilities(
    intensity_g: float,
    taxonomy: str,
    fragility_lookup: dict[str, dict[str, tuple[float, float]]],
) -> dict[str, float]:
    parameters = fragility_lookup[taxonomy]
    exceedance: dict[str, float] = {}

    for state in ["DS1", "DS2", "DS3", "DS4"]:
        median, beta = parameters[state]
        if intensity_g <= 0:
            probability = 0.0
        else:
            z_score = (log(intensity_g) - log(median)) / beta
            probability = float(np.clip(normal_cdf(z_score), 0.0, 1.0))
        exceedance[state] = probability

    exceedance["DS2"] = min(exceedance["DS1"], exceedance["DS2"])
    exceedance["DS3"] = min(exceedance["DS2"], exceedance["DS3"])
    exceedance["DS4"] = min(exceedance["DS3"], exceedance["DS4"])

    probabilities = {
        "DS0": max(0.0, 1.0 - exceedance["DS1"]),
        "DS1": max(0.0, exceedance["DS1"] - exceedance["DS2"]),
        "DS2": max(0.0, exceedance["DS2"] - exceedance["DS3"]),
        "DS3": max(0.0, exceedance["DS3"] - exceedance["DS4"]),
        "DS4": max(0.0, exceedance["DS4"]),
    }

    total = sum(probabilities.values())
    if total <= 0:
        return {"DS0": 1.0, "DS1": 0.0, "DS2": 0.0, "DS3": 0.0, "DS4": 0.0}
    return {state: value / total for state, value in probabilities.items()}


def expected_debris_volume(
    footprint_area_m2: float,
    levels: float,
    taxonomy: str,
    probabilities: dict[str, float],
    debris_lookup: dict[str, dict[str, float]],
) -> tuple[float, float]:
    floor_area_m2 = max(0.0, footprint_area_m2) * max(1.0, levels)
    rates = debris_lookup[taxonomy]
    expected_rate = sum(probabilities[state] * rates[state] for state in DAMAGE_STATES)
    return floor_area_m2 * expected_rate, expected_rate


# =============================================================================
# 5. OSM DOWNLOAD AND ROUTE FUNCTIONS
# =============================================================================


@st.cache_data(ttl=3600, show_spinner=False)
def geocode_place(place_name: str) -> tuple[float, float]:
    lat, lon = ox.geocoder.geocode(place_name)
    return float(lat), float(lon)


@st.cache_resource(ttl=3600, show_spinner=False)
def download_graph_cached(
    bbox: tuple[float, float, float, float],
    server_url: str,
) -> nx.MultiDiGraph:
    ox.settings.overpass_url = server_url
    try:
        return ox.graph_from_bbox(
            bbox=bbox,
            network_type="all",
            simplify=True,
            retain_all=True,
            truncate_by_edge=True,
        )
    except TypeError:
        west, south, east, north = bbox
        return ox.graph_from_bbox(
            north, south, east, west,
            network_type="all",
            simplify=True,
            retain_all=True,
            truncate_by_edge=True,
        )


@st.cache_data(ttl=3600, show_spinner=False)
def download_buildings_bbox_cached(
    bbox: tuple[float, float, float, float],
    target_crs_text: str,
    server_url: str,
) -> gpd.GeoDataFrame:
    ox.settings.overpass_url = server_url
    try:
        buildings = ox.features_from_bbox(bbox=bbox, tags={"building": True})
    except TypeError:
        west, south, east, north = bbox
        buildings = ox.geometries_from_bbox(
            north, south, east, west, tags={"building": True}
        )
    if buildings.empty:
        return gpd.GeoDataFrame(geometry=[], crs=target_crs_text)
    return ox.projection.project_gdf(buildings, to_crs=target_crs_text)


def choose_shortest_edge_data(graph: nx.MultiGraph, u: Any, v: Any) -> dict[str, Any] | None:
    edge_bundle = graph.get_edge_data(u, v)
    if not edge_bundle:
        return None
    return min(
        edge_bundle.values(),
        key=lambda data: float(data.get("length", np.inf)),
    )


def route_geometry_from_path(
    graph: nx.MultiGraph,
    route: list[Any],
) -> tuple[Any, float]:
    geometries = []
    route_length = 0.0

    for route_u, route_v in zip(route[:-1], route[1:]):
        edge_data = choose_shortest_edge_data(graph, route_u, route_v)
        if edge_data is None:
            continue

        route_length += float(edge_data.get("length", 0.0))
        geometry = edge_data.get("geometry")
        if geometry is None:
            u_data = graph.nodes[route_u]
            v_data = graph.nodes[route_v]
            geometry = LineString(
                [(u_data["x"], u_data["y"]), (v_data["x"], v_data["y"])]
            )
        geometries.append(geometry)

    if not geometries:
        raise nx.NetworkXNoPath("No geometries found for shortest path.")

    united = unary_union(geometries)
    if united.geom_type == "LineString":
        merged = united
    elif united.geom_type == "MultiLineString":
        merged = linemerge(united)
    else:
        line_parts = [
            geometry
            for geometry in getattr(united, "geoms", [])
            if geometry.geom_type in {"LineString", "MultiLineString"}
        ]
        if not line_parts:
            raise nx.NetworkXNoPath("Merged route geometry contains no linework.")
        line_union = unary_union(line_parts)
        merged = line_union if line_union.geom_type == "LineString" else linemerge(line_union)

    if merged.is_empty:
        raise nx.NetworkXNoPath("Merged route geometry is empty.")

    geometry_length = float(merged.length)
    return merged, geometry_length if geometry_length > 0 else route_length


def build_selected_network(
    nodes_gdf: gpd.GeoDataFrame,
    graph_projected: nx.MultiDiGraph,
    physical_pairs: list[tuple[int, int]],
) -> gpd.GeoDataFrame:
    if hasattr(ox, "convert") and hasattr(ox.convert, "to_undirected"):
        graph_undirected = ox.convert.to_undirected(graph_projected)
    else:
        graph_undirected = ox.utils_graph.get_undirected(graph_projected)
    nearest_nodes = ox.distance.nearest_nodes(
        graph_projected,
        X=nodes_gdf.geometry.x.to_numpy(),
        Y=nodes_gdf.geometry.y.to_numpy(),
    )
    nearest_nodes = np.atleast_1d(nearest_nodes)

    edge_records: list[dict[str, Any]] = []
    for road_number, (u_idx, v_idx) in enumerate(physical_pairs):
        u_label, v_label = f"i{u_idx}", f"i{v_idx}"
        p1 = nodes_gdf.loc[u_label].geometry
        p2 = nodes_gdf.loc[v_label].geometry
        fallback_geometry = LineString([p1, p2])
        route_geometry = fallback_geometry
        route_length_m = float(fallback_geometry.length)
        route_source = "straight_fallback"

        try:
            origin = nearest_nodes[u_idx]
            destination = nearest_nodes[v_idx]
            if origin != destination:
                route = nx.shortest_path(
                    graph_undirected,
                    origin,
                    destination,
                    weight="length",
                )
                route_geometry, route_length_m = route_geometry_from_path(
                    graph_undirected,
                    route,
                )
                route_source = "osm_shortest_path"
            else:
                route_source = "same_osm_node_fallback"
        except (nx.NetworkXNoPath, nx.NodeNotFound, ValueError, TypeError):
            route_source = "straight_fallback"

        edge_records.append(
            {
                "road_id": f"r{road_number}",
                "u": u_label,
                "v": v_label,
                "length_m": route_length_m,
                "route_source": route_source,
                "geometry": route_geometry,
            }
        )

    return gpd.GeoDataFrame(edge_records, geometry="geometry", crs=nodes_gdf.crs)


# =============================================================================
# 6. BUILDING PROCESSING AND DEBRIS ALLOCATION
# =============================================================================


def prepare_buildings(
    buildings: gpd.GeoDataFrame,
    fragility_df: pd.DataFrame,
    debris_rates_df: pd.DataFrame,
) -> gpd.GeoDataFrame:
    if buildings.empty:
        return gpd.GeoDataFrame(geometry=[], crs=buildings.crs)

    valid = buildings[buildings.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if valid.empty:
        return valid

    try:
        valid.geometry = valid.geometry.make_valid()
    except AttributeError:
        valid.geometry = valid.geometry.buffer(0)

    valid = valid[valid.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    valid = valid[~valid.geometry.is_empty & valid.geometry.notna()].copy()
    valid["b_id"] = [f"b{index}" for index in range(len(valid))]
    valid["footprint_area_m2"] = valid.geometry.area.astype(float)
    valid = valid[valid["footprint_area_m2"] > 1.0].copy()

    level_results = valid.apply(infer_levels, axis=1)
    valid["levels"] = [result[0] for result in level_results]
    valid["levels_source"] = [result[1] for result in level_results]

    taxonomy_results = valid.apply(
        lambda row: infer_taxonomy(row, float(row["levels"])),
        axis=1,
    )
    valid["taxonomy"] = [result[0] for result in taxonomy_results]
    valid["taxonomy_source"] = [result[1] for result in taxonomy_results]

    allowed_taxonomies = set(fragility_df["taxonomy"]) & set(debris_rates_df["taxonomy"])
    fallback_taxonomy = "MUR_M" if "MUR_M" in allowed_taxonomies else sorted(allowed_taxonomies)[0]
    valid.loc[~valid["taxonomy"].isin(allowed_taxonomies), "taxonomy"] = fallback_taxonomy

    fragility_lookup = fragility_lookup_from_df(fragility_df)
    debris_lookup = debris_lookup_from_df(debris_rates_df)

    probability_rows: list[dict[str, float]] = []
    expected_volumes: list[float] = []
    expected_rates: list[float] = []
    damage_indices: list[float] = []

    for _, row in valid.iterrows():
        taxonomy = str(row["taxonomy"])
        probabilities = damage_state_probabilities(
            FIXED_PGA_G,
            taxonomy,
            fragility_lookup,
        )
        expected_volume, expected_rate = expected_debris_volume(
            footprint_area_m2=float(row["footprint_area_m2"]),
            levels=float(row["levels"]),
            taxonomy=taxonomy,
            probabilities=probabilities,
            debris_lookup=debris_lookup,
        )

        probability_rows.append(probabilities)
        expected_volumes.append(expected_volume)
        expected_rates.append(expected_rate)
        damage_indices.append(
            sum(index * probabilities[f"DS{index}"] for index in range(5))
        )

    probability_df = pd.DataFrame(probability_rows, index=valid.index)
    for state in DAMAGE_STATES:
        valid[f"P_{state}"] = probability_df[state]

    valid["total_floor_area_m2"] = valid["footprint_area_m2"] * valid["levels"]
    valid["expected_debris_rate_m3_m2"] = expected_rates
    valid["expected_debris_m3"] = expected_volumes
    valid["expected_damage_index"] = damage_indices

    return valid


def allocate_debris_to_roads(
    buildings: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    roads_result = roads.copy()
    roads_result["Q_m3"] = 0.0
    roads_result["blocked"] = 0
    roads_result["affected_buildings"] = 0

    allocation_columns = [
        "b_id",
        "road_id",
        "intersection_area_m2",
        "normalized_weight",
        "coverage_ratio",
        "allocated_debris_m3",
    ]

    if buildings.empty or roads.empty:
        return pd.DataFrame(columns=allocation_columns), roads_result

    buffered_roads = roads[["road_id", "geometry"]].copy()
    buffered_roads.geometry = buffered_roads.geometry.buffer(ROAD_INFLUENCE_BUFFER_M)
    street_union = union_all_geometries(buffered_roads.geometry)

    building_subset = buildings[
        ["b_id", "footprint_area_m2", "expected_debris_m3", "geometry"]
    ].copy()
    building_subset["unique_covered_area_m2"] = building_subset.geometry.apply(
        lambda geometry: float(geometry.intersection(street_union).area)
        if not geometry.is_empty
        else 0.0
    )
    building_subset["coverage_ratio"] = np.where(
        building_subset["footprint_area_m2"] > 0,
        np.minimum(
            1.0,
            building_subset["unique_covered_area_m2"]
            / building_subset["footprint_area_m2"],
        ),
        0.0,
    )
    building_subset["debris_available_to_roads_m3"] = (
        building_subset["expected_debris_m3"]
        * STREET_DEPOSITION_FACTOR
        * building_subset["coverage_ratio"]
    )

    intersections = gpd.overlay(
        building_subset[
            [
                "b_id",
                "footprint_area_m2",
                "coverage_ratio",
                "debris_available_to_roads_m3",
                "geometry",
            ]
        ],
        buffered_roads[["road_id", "geometry"]],
        how="intersection",
        keep_geom_type=False,
    )

    if intersections.empty:
        return pd.DataFrame(columns=allocation_columns), roads_result

    intersections["intersection_area_m2"] = intersections.geometry.area.astype(float)
    intersections = intersections[intersections["intersection_area_m2"] > 0].copy()
    if intersections.empty:
        return pd.DataFrame(columns=allocation_columns), roads_result

    weight_sums = intersections.groupby("b_id")["intersection_area_m2"].transform("sum")
    intersections["normalized_weight"] = np.where(
        weight_sums > 0,
        intersections["intersection_area_m2"] / weight_sums,
        0.0,
    )
    intersections["allocated_debris_m3"] = (
        intersections["normalized_weight"]
        * intersections["debris_available_to_roads_m3"]
    )

    debris_by_road = intersections.groupby("road_id")["allocated_debris_m3"].sum()
    buildings_by_road = intersections.groupby("road_id")["b_id"].nunique()

    roads_result["Q_m3"] = roads_result["road_id"].map(debris_by_road).fillna(0.0)
    roads_result["affected_buildings"] = (
        roads_result["road_id"].map(buildings_by_road).fillna(0).astype(int)
    )
    roads_result["blocked"] = (
        roads_result["Q_m3"] >= MIN_DEBRIS_THRESHOLD_M3
    ).astype(int)

    allocation_df = intersections[
        [
            "b_id",
            "road_id",
            "intersection_area_m2",
            "normalized_weight",
            "coverage_ratio",
            "allocated_debris_m3",
        ]
    ].copy()

    return allocation_df, roads_result


# =============================================================================
# 7. OUTPUT BUILDERS
# =============================================================================


def build_excel_output(
    coord_df: pd.DataFrame,
    roads_gdf: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
    allocation_df: pd.DataFrame,
    fragility_df: pd.DataFrame,
    debris_rates_df: pd.DataFrame,
    parameter_source: str,
) -> bytes:
    roads_df = roads_gdf.drop(columns="geometry").copy()
    roads_df["Q_m3"] = roads_df["Q_m3"].round(3)
    roads_df["length_m"] = roads_df["length_m"].round(3)

    edge_rows: list[dict[str, Any]] = []
    debris_rows: list[dict[str, Any]] = []
    for _, road in roads_df.iterrows():
        # Add bidirectional edges for GAMS
        edge_rows.append({"Node_i": road["u"], "Node_j": road["v"]})
        edge_rows.append({"Node_i": road["v"], "Node_j": road["u"]})

        # Add debris for blocked roads only
        if int(road["blocked"]) == 1:
            q_value = round(float(road["Q_m3"]), 2)
            debris_rows.append({"Node_i": road["u"], "Node_j": road["v"], "Volume_m3": q_value})
            debris_rows.append({"Node_i": road["v"], "Node_j": road["u"], "Volume_m3": q_value})

    edges_df = pd.DataFrame(edge_rows, columns=["Node_i", "Node_j"])
    debris_df = pd.DataFrame(debris_rows, columns=["Node_i", "Node_j", "Volume_m3"])
    
    # Filter Coord DataFrame to only include Node, X, Y for GAMS
    coord_gams_df = coord_df[["Node", "X", "Y"]].copy()

    output_buffer = io.BytesIO()
    with pd.ExcelWriter(output_buffer, engine="xlsxwriter") as writer:
        coord_gams_df.to_excel(writer, sheet_name="Coord", index=False)
        edges_df.to_excel(writer, sheet_name="Edges", index=False)
        debris_df.to_excel(writer, sheet_name="Debris", index=False)

    return output_buffer.getvalue()


def build_graph_image(
    nodes_gdf: gpd.GeoDataFrame,
    roads_gdf: gpd.GeoDataFrame,
) -> bytes:
    figure, axis = plt.subplots(figsize=(15, 15), facecolor="white")

    # رسم یال‌ها به صورت خط صاف و مستقیم
    for _, road in roads_gdf.iterrows():
        u_node, v_node = road["u"], road["v"]
        x1, y1 = nodes_gdf.loc[u_node].geometry.x, nodes_gdf.loc[u_node].geometry.y
        x2, y2 = nodes_gdf.loc[v_node].geometry.x, nodes_gdf.loc[v_node].geometry.y
        
        q_value = float(road["Q_m3"])
        is_blocked = int(road["blocked"]) == 1
        
        color = "red" if is_blocked else "gray"
        lw = 2.5 if is_blocked else 1.2
        axis.plot([x1, x2], [y1, y2], color=color, linewidth=lw, zorder=1, alpha=0.9 if is_blocked else 0.5)
        
        if is_blocked:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            axis.text(mid_x, mid_y, f"Q={q_value:.2f}", color="black", fontsize=10, fontweight="bold",
                      bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", boxstyle="round,pad=0.3"), zorder=4)

    # رسم گره‌ها
    for node_id, row in nodes_gdf.iterrows():
        x_coord, y_coord = row.geometry.x, row.geometry.y
        node_number = str(node_id).replace("i", "")
        if node_id == "i0":
            axis.scatter(x_coord, y_coord, s=700, color="#2ca02c", edgecolor="black", linewidth=2, zorder=3)
            axis.text(x_coord, y_coord, "0 (Depot)", color="white", fontsize=9, fontweight="bold", ha="center", va="center", zorder=4)
        else:
            axis.scatter(x_coord, y_coord, s=700, color="#1f77b4", edgecolor="black", linewidth=2, zorder=3)
            axis.text(x_coord, y_coord, node_number, color="white", fontsize=12, fontweight="bold", ha="center", va="center", zorder=4)

    axis.axis("off")
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_title(
        f"Network with Expected Debris Volume — {SCENARIO_NAME}",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )

    image_buffer = io.BytesIO()
    figure.savefig(image_buffer, format="png", dpi=300, bbox_inches="tight")
    plt.close(figure)
    return image_buffer.getvalue()


def build_satellite_image(
    nodes_gdf: gpd.GeoDataFrame,
    roads_gdf: gpd.GeoDataFrame,
) -> bytes:
    figure, axis = plt.subplots(figsize=(15, 15))
    nodes_3857 = nodes_gdf.to_crs(epsg=3857)

    # رسم خطوط کاملا صاف و مستقیم روی عکس ماهواره‌ای
    for _, road in roads_gdf.iterrows():
        u_node, v_node = road["u"], road["v"]
        x1, y1 = nodes_3857.loc[u_node].geometry.x, nodes_3857.loc[u_node].geometry.y
        x2, y2 = nodes_3857.loc[v_node].geometry.x, nodes_3857.loc[v_node].geometry.y
        
        is_blocked = int(road["blocked"]) == 1
        # خط زیرین ضخیم سفید برای وضوح روی عکس
        axis.plot([x1, x2], [y1, y2], color="white", linewidth=4, zorder=1, solid_capstyle='round')
        # خط اصلی مشکی یا قرمز
        color = "red" if is_blocked else "black"
        lw = 2.5 if is_blocked else 1.5
        axis.plot([x1, x2], [y1, y2], color=color, linewidth=lw, zorder=2, alpha=0.9)

    nodes_3857.plot(ax=axis, markersize=150, color="yellow", edgecolor="black", linewidth=2, zorder=4)
    for node_id, row in nodes_3857.iterrows():
        axis.text(row.geometry.x, row.geometry.y, str(node_id).replace("i", ""), fontsize=10, color="black", fontweight="bold", ha="center", va="center", zorder=5)

    try:
        ctx.add_basemap(axis, source=ctx.providers.Esri.WorldImagery)
    except Exception:
        axis.set_facecolor("lightgray")

    axis.axis("off")
    axis.set_title(
        f"Satellite Network Map — {SCENARIO_NAME}",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )

    image_buffer = io.BytesIO()
    figure.savefig(image_buffer, format="png", dpi=300, bbox_inches="tight")
    plt.close(figure)
    return image_buffer.getvalue()


# =============================================================================
# 8. USER INTERFACE
# =============================================================================

ensure_session_state()
fragility_parameters, debris_rates, parameter_source_text = load_parameter_tables()

with st.sidebar:
    st.header("سناریوی ثابت")
    st.metric("PGA ثابت", f"{FIXED_PGA_G:.2f} g")
    st.write(f"**نام سناریو:** `{SCENARIO_NAME}`")
    st.write(f"**بافر اثر خیابان:** {ROAD_INFLUENCE_BUFFER_M:.0f} m")
    st.write(f"**ضریب ورود آوار به خیابان:** {STREET_DEPOSITION_FACTOR:.2f}")
    st.write(f"**آستانه آوار:** {MIN_DEBRIS_THRESHOLD_M3:.1f} m³")
    st.caption(parameter_source_text)

    with st.expander("جدول شکنندگی مورد استفاده"):
        st.dataframe(fragility_parameters, use_container_width=True, hide_index=True)
    with st.expander("جدول نرخ آوار مورد استفاده"):
        st.dataframe(debris_rates, use_container_width=True, hide_index=True)

st.write(
    "1. محله را انتخاب کنید. "
    "2. با ابزار Marker روی نقشه گره‌ها را ثبت کنید. "
    "3. اتصالات را بررسی یا دستی ویرایش کنید. "
    "4. دکمه پردازش را بزنید و خروجی‌ها را دریافت کنید."
)

servers = {
    "Germany (پایدار)": "https://overpass-api.de/api/interpreter",
    "Kumi Systems": "https://overpass.kumi.systems/api/interpreter",
}
selected_server_name = st.selectbox("انتخاب سرور دانلود:", list(servers.keys()))
selected_server_url = servers[selected_server_name]
ox.settings.overpass_url = selected_server_url

berlin_districts = [
    "Kreuzberg, Berlin, Germany",
    "Mitte, Berlin, Germany",
    "Friedrichshain, Berlin, Germany",
    "Prenzlauer Berg, Berlin, Germany",
    "Charlottenburg, Berlin, Germany",
    "Neukölln, Berlin, Germany",
    "Tiergarten, Berlin, Germany",
    "Wedding, Berlin, Germany",
    "Wilmersdorf, Berlin, Germany",
]
place_name = st.selectbox("انتخاب محله در برلین:", berlin_districts)

try:
    city_lat, city_lon = geocode_place(place_name)
except Exception:
    city_lat, city_lon = 52.5076, 13.3930

map_object = folium.Map(
    location=[city_lat, city_lon],
    zoom_start=15,
    tiles="Esri.WorldImagery",
)
Draw(
    export=False,
    draw_options={
        "polygon": False,
        "polyline": False,
        "rectangle": False,
        "circle": False,
        "circlemarker": False,
        "marker": True,
    },
    edit_options={"edit": False, "remove": False},
).add_to(map_object)

if st.session_state.map_layers:
    map_nodes, map_edges, blocked_road_ids = st.session_state.map_layers
    for road_id, edge_coordinates in map_edges:
        edge_color = "red" if road_id in blocked_road_ids else "yellow"
        folium.PolyLine(
            edge_coordinates,
            color=edge_color,
            weight=5,
            opacity=0.85,
            tooltip=road_id,
        ).add_to(map_object)

    for node_id, coordinates in map_nodes.items():
        folium.CircleMarker(
            location=coordinates,
            radius=8,
            color="black",
            fill=True,
            fill_color="yellow",
            fill_opacity=1,
        ).add_to(map_object)
        add_numbered_marker(
            map_object,
            coordinates[0],
            coordinates[1],
            int(node_id.replace("i", "")),
            color_bg="rgba(255,255,0,0.8)",
            border_color="black",
            font_size="10pt",
            size=22,
        )
else:
    for point_number, (latitude, longitude) in enumerate(st.session_state.drawn_points):
        add_numbered_marker(map_object, latitude, longitude, point_number)

map_output = st_folium(
    map_object,
    width=1000,
    height=600,
    key=f"map_{st.session_state.map_reset_counter}",
)

if map_output and map_output.get("all_drawings"):
    new_point_added = False
    for drawing in map_output["all_drawings"]:
        if drawing.get("geometry", {}).get("type") != "Point":
            continue
        longitude, latitude = drawing["geometry"]["coordinates"]
        is_existing = any(
            abs(existing_lat - latitude) < 0.00001
            and abs(existing_lon - longitude) < 0.00001
            for existing_lat, existing_lon in st.session_state.drawn_points
        )
        if not is_existing and len(st.session_state.drawn_points) < MAX_POINTS:
            st.session_state.drawn_points.append((latitude, longitude))
            new_point_added = True

    if new_point_added:
        number_of_points = len(st.session_state.drawn_points)
        if number_of_points >= 2:
            new_edge = f"{number_of_points - 2}-{number_of_points - 1}"
            current_edges = [
                edge.strip()
                for edge in st.session_state.custom_edges_str.split(",")
                if edge.strip()
            ]
            if new_edge not in current_edges:
                st.session_state.custom_edges_str += (
                    f", {new_edge}" if st.session_state.custom_edges_str else new_edge
                )
        st.session_state.map_reset_counter += 1
        st.rerun()

st.write("---")

if st.session_state.drawn_points:
    st.success(
        f"✅ تعداد گره‌ها: **{len(st.session_state.drawn_points)}** "
        f"(شماره 0 تا {len(st.session_state.drawn_points) - 1})"
    )

    clear_column, delete_select_column, delete_button_column = st.columns([2, 2, 1])
    with clear_column:
        if st.button("🧹 پاک کردن همه نقاط"):
            st.session_state.drawn_points = []
            st.session_state.map_layers = None
            st.session_state.custom_edges_str = ""
            st.session_state.excel_data = b""
            st.session_state.graph_img_data = b""
            st.session_state.sat_img_data = b""
            st.session_state.stats = ""
            st.session_state.map_reset_counter += 1
            st.rerun()

    with delete_select_column:
        delete_index = st.selectbox(
            "انتخاب گره برای حذف:",
            options=list(range(len(st.session_state.drawn_points))),
            index=len(st.session_state.drawn_points) - 1,
        )

    with delete_button_column:
        st.write("")
        if st.button("🗑️ حذف گره"):
            st.session_state.drawn_points.pop(delete_index)
            old_pairs = parse_custom_edges(
                st.session_state.custom_edges_str,
                len(st.session_state.drawn_points) + 1,
            )
            new_pairs: list[tuple[int, int]] = []
            for old_u, old_v in old_pairs:
                if old_u == delete_index or old_v == delete_index:
                    continue
                new_u = old_u - 1 if old_u > delete_index else old_u
                new_v = old_v - 1 if old_v > delete_index else old_v
                new_pairs.append((new_u, new_v))
            st.session_state.custom_edges_str = ", ".join(
                f"{u}-{v}" for u, v in new_pairs
            )
            st.session_state.map_layers = None
            st.session_state.map_reset_counter += 1
            st.rerun()

st.write("### تعریف اتصالات شبکه")

if st.button("🛣️ بررسی اتصال مستقیم خیابان‌ها (OSM)"):
    if len(st.session_state.drawn_points) < 2:
        st.warning("حداقل دو گره ثبت کنید.")
    else:
        try:
            with st.spinner("در حال بررسی خیابان‌های OSM..."):
                latitudes = [point[0] for point in st.session_state.drawn_points]
                longitudes = [point[1] for point in st.session_state.drawn_points]
                bbox = (
                    min(longitudes) - BBOX_PADDING_DEG,
                    min(latitudes) - BBOX_PADDING_DEG,
                    max(longitudes) + BBOX_PADDING_DEG,
                    max(latitudes) + BBOX_PADDING_DEG,
                )

                graph = download_graph_cached(bbox, selected_server_url)
                point_records = [
                    {"node": f"i{index}", "geometry": Point(lon, lat)}
                    for index, (lat, lon) in enumerate(st.session_state.drawn_points)
                ]
                point_gdf = gpd.GeoDataFrame(point_records, crs="EPSG:4326").set_index("node")
                point_gdf = ox.projection.project_gdf(point_gdf)
                graph_projected = ox.project_graph(graph, to_crs=point_gdf.crs)

                nearest_edges = ox.distance.nearest_edges(
                    graph_projected,
                    X=point_gdf.geometry.x.to_numpy(),
                    Y=point_gdf.geometry.y.to_numpy(),
                )
                nearest_edges = [tuple(edge) for edge in np.atleast_2d(nearest_edges)]

                detected_edges: set[tuple[int, int]] = set()
                number_of_points = len(point_gdf)
                for i in range(number_of_points):
                    for j in range(i + 1, number_of_points):
                        point_distance = float(
                            point_gdf.iloc[i].geometry.distance(point_gdf.iloc[j].geometry)
                        )
                        if point_distance > MAX_CONNECTION_DISTANCE_M:
                            continue

                        edge_i = nearest_edges[i]
                        edge_j = nearest_edges[j]
                        nodes_i = {edge_i[0], edge_i[1]}
                        nodes_j = {edge_j[0], edge_j[1]}
                        if edge_i == edge_j or nodes_i.intersection(nodes_j):
                            detected_edges.add((i, j))

                if detected_edges:
                    st.session_state.custom_edges_str = ", ".join(
                        f"{u}-{v}" for u, v in sorted(detected_edges)
                    )
                else:
                    st.warning("اتصال مستقیم نزدیک بین نقاط پیدا نشد؛ اتصالات را دستی وارد کنید.")
                st.rerun()
        except Exception as error:
            st.error(f"خطا در بررسی اتصال خیابان‌ها: {error}")

st.text_area(
    "اتصالات فیزیکی را وارد کنید (مثال: 0-1, 1-2, 1-3):",
    key="custom_edges_str",
)

button_column, excel_column, graph_column, satellite_column = st.columns(4)
with button_column:
    process_clicked = st.button("Process Network", type="primary")
with excel_column:
    st.download_button(
        label="📥 Excel",
        data=st.session_state.excel_data,
        file_name="Fixed_Scenario_Debris_Network.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        disabled=st.session_state.excel_data == b"",
    )
with graph_column:
    st.download_button(
        label="🖼️ Graph Image",
        data=st.session_state.graph_img_data,
        file_name="Fixed_Scenario_Network_Graph.png",
        mime="image/png",
        disabled=st.session_state.graph_img_data == b"",
    )
with satellite_column:
    st.download_button(
        label="🛰️ Satellite Map",
        data=st.session_state.sat_img_data,
        file_name="Fixed_Scenario_Satellite_Map.png",
        mime="image/png",
        disabled=st.session_state.sat_img_data == b"",
    )

if st.session_state.stats:
    st.info(st.session_state.stats)


# =============================================================================
# 9. MAIN PROCESS
# =============================================================================

if process_clicked:
    selected_points = st.session_state.drawn_points
    if len(selected_points) < 2:
        st.error("لطفاً حداقل دو گره روی نقشه قرار دهید.")
        st.stop()

    physical_pairs = parse_custom_edges(
        st.session_state.custom_edges_str,
        len(selected_points),
    )
    if not physical_pairs:
        st.error("هیچ اتصال معتبر وارد نشده است.")
        st.stop()

    try:
        progress = st.progress(0, text="شروع پردازش...")

        node_records = [
            {"Node": f"i{index}", "geometry": Point(longitude, latitude)}
            for index, (latitude, longitude) in enumerate(selected_points)
        ]
        nodes_wgs84 = gpd.GeoDataFrame(node_records, crs="EPSG:4326").set_index("Node")
        nodes_gdf = ox.projection.project_gdf(nodes_wgs84)
        nodes_gdf["x"] = nodes_gdf.geometry.x
        nodes_gdf["y"] = nodes_gdf.geometry.y

        coord_df = pd.DataFrame(
            {
                "Node": nodes_gdf.index,
                "X": nodes_gdf["x"].to_numpy() / 1000.0,
                "Y": nodes_gdf["y"].to_numpy() / 1000.0,
                "Longitude": nodes_wgs84.geometry.x.to_numpy(),
                "Latitude": nodes_wgs84.geometry.y.to_numpy(),
            }
        )

        progress.progress(20, text="🛣️ دانلود شبکه خیابان‌ها و ساخت مسیرهای واقعی...")
        latitudes = [point[0] for point in selected_points]
        longitudes = [point[1] for point in selected_points]
        bbox = (
            min(longitudes) - BBOX_PADDING_DEG,
            min(latitudes) - BBOX_PADDING_DEG,
            max(longitudes) + BBOX_PADDING_DEG,
            max(latitudes) + BBOX_PADDING_DEG,
        )

        osm_graph = download_graph_cached(bbox, selected_server_url)
        osm_graph_projected = ox.project_graph(osm_graph, to_crs=nodes_gdf.crs)
        roads_gdf = build_selected_network(
            nodes_gdf,
            osm_graph_projected,
            physical_pairs,
        )
        if roads_gdf.empty:
            st.error("هیچ مسیر معتبری ساخته نشد.")
            st.stop()

        progress.progress(45, text="🏢 دانلود و آماده‌سازی ساختمان‌های OSM...")
        raw_buildings = download_buildings_bbox_cached(
            bbox,
            str(nodes_gdf.crs),
            selected_server_url,
        )
        processed_buildings = prepare_buildings(
            raw_buildings,
            fragility_parameters,
            debris_rates,
        )

        progress.progress(70, text="🏗️ محاسبه خرابی و تخصیص آوار به خیابان‌ها...")
        allocation_df, roads_with_debris = allocate_debris_to_roads(
            processed_buildings,
            roads_gdf,
        )

        progress.progress(85, text="📊 ساخت Excel و تصاویر خروجی...")
        st.session_state.excel_data = build_excel_output(
            coord_df,
            roads_with_debris,
            processed_buildings,
            allocation_df,
            fragility_parameters,
            debris_rates,
            parameter_source_text,
        )
        st.session_state.graph_img_data = build_graph_image(
            nodes_gdf,
            roads_with_debris,
        )
        st.session_state.sat_img_data = build_satellite_image(
            nodes_gdf,
            roads_with_debris,
        )

        nodes_4326 = nodes_gdf.to_crs(epsg=4326)
        roads_4326 = roads_with_debris.to_crs(epsg=4326)
        map_nodes = {
            node_id: [row.geometry.y, row.geometry.x]
            for node_id, row in nodes_4326.iterrows()
        }
        map_edges: list[tuple[str, list[tuple[float, float]]]] = []
        for _, road in roads_4326.iterrows():
            geometry = road.geometry
            line_parts = list(geometry.geoms) if geometry.geom_type == "MultiLineString" else [geometry]
            for line in line_parts:
                map_edges.append(
                    (
                        str(road["road_id"]),
                        [(latitude, longitude) for longitude, latitude in line.coords],
                    )
                )

        blocked_road_ids = set(
            roads_with_debris.loc[
                roads_with_debris["blocked"] == 1,
                "road_id",
            ].astype(str)
        )
        st.session_state.map_layers = (map_nodes, map_edges, blocked_road_ids)

        total_expected_building_debris = (
            float(processed_buildings["expected_debris_m3"].sum())
            if not processed_buildings.empty
            else 0.0
        )
        total_road_debris = float(roads_with_debris["Q_m3"].sum())
        blocked_count = int(roads_with_debris["blocked"].sum())
        fallback_route_count = int(
            roads_with_debris["route_source"].ne("osm_shortest_path").sum()
        )

        st.session_state.stats = (
            f"✅ سناریو: {SCENARIO_NAME} | PGA={FIXED_PGA_G:.2f}g\n"
            f"گره‌ها: {len(coord_df)} | خیابان‌های فیزیکی: {len(roads_with_debris)} | "
            f"ساختمان‌ها: {len(processed_buildings)}\n"
            f"حجم مورد انتظار آوار ساختمان‌ها: {total_expected_building_debris:,.2f} m³ | "
            f"آوار تخصیص‌یافته به خیابان‌ها: {total_road_debris:,.2f} m³ | "
            f"خیابان‌های بالای آستانه: {blocked_count}\n"
            f"مسیرهای دارای خط مستقیم جایگزین: {fallback_route_count}"
        )

        progress.progress(100, text="✅ تکمیل شد")
        st.rerun()

    except Exception as error:
        st.error(f"خطا در پردازش: {error}")
        st.exception(error)