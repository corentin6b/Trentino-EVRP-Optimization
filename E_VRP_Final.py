import os
# Mac (M1/M2/M3) fix for Protobuf implementation conflict
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import pandas as pd
import numpy as np
import folium
import requests
from streamlit_folium import st_folium
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

st.set_page_config(page_title="Trentino E-VRP PRO", layout="wide")
st.title("⚡ E-CVRP Optimization – Trentino Eco-Laundry (OSRM Routing)")

# --- 1. DATA INGESTION FROM CSV ---
@st.cache_data
def load_data_from_csv(filepath):
    nodes = {}
    try:
        df = pd.read_csv(filepath, sep=None, engine='python', encoding='latin1')
        df.columns = df.columns.str.strip()

        required_cols = ['Nome', 'Latitudine', 'Longitudine', 'Altitudine', 'Tipo', 'Domanda_kg']
        for col in required_cols:
            if col not in df.columns:
                st.error(f"❌ Missing required column in CSV: '{col}'")
                st.stop()

        for _, row in df.iterrows():
            name = str(row['Nome'])
            if pd.isna(name) or name.strip() == "":
                continue
            nodes[name] = {
                "lat": float(row['Latitudine']),
                "lon": float(row['Longitudine']),
                "alt": float(row['Altitudine']),
                "type": str(row['Tipo']).strip().lower(),
                "demand": float(row['Domanda_kg']) if pd.notna(row['Domanda_kg']) else 0.0
            }
        return nodes
    except FileNotFoundError:
        st.error(f"❌ File {filepath} not found. Please ensure it is in the script directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error while parsing CSV file: {e}")
        st.stop()

CSV_FILE = "Base_VRP.csv"
nodes_data = load_data_from_csv(CSV_FILE)

if not nodes_data:
    st.stop()

node_names = list(nodes_data.keys())
num_nodes = len(node_names)

# --- 2. INTERACTIVE HYPERPARAMETERS (SIDEBAR) ---
st.sidebar.header("⚙️ Cost Function & Fleet Parameters")
fixed_cost = st.sidebar.slider("Vehicle Fixed Cost (€/day)", 50, 200, 120)
driver_rate = st.sidebar.slider("Driver Hourly Wage (€/h)", 15, 45, 30)
kwh_cost = st.sidebar.slider("Neogy Fast Charging Cost (€/kWh)", 0.40, 0.90, 0.79)
max_fleet_size = st.sidebar.slider("Maximum Available Vans in Fleet", 10, 25, 18)

van_capacity = st.sidebar.slider("Linen Payload Capacity (kg)", 800, 1600, 1200)
battery_cap_kwh = st.sidebar.slider("Usable Battery Capacity (kWh)", 40, 120, 69)

# --- 3. MATRIX GENERATION & ADVANCED EV PHYSICAL MODEL ---
@st.cache_data
def build_matrices_with_osrm(data):
    n = len(data)
    dist_matrix = np.zeros((n, n), dtype=int)
    time_matrix = np.zeros((n, n), dtype=int)
    energy_matrix = np.zeros((n, n), dtype=int)
    
    coords = ";".join([f"{v['lon']},{v['lat']}" for v in data.values()])
    url = f"http://router.project-osrm.org/table/v1/driving/{coords}?annotations=distance,duration"
    
    try:
        response = requests.get(url).json()
        osrm_distances = response['distances']
        osrm_durations = response['durations']
    except Exception as e:
        st.error(f"OSRM Connection Error: {e}. Please verify your internet connection.")
        st.stop()

    names = list(data.keys())
    base_consumption = 244
    
    for i in range(n):
        for j in range(n):
            if i == j: continue
            
            n1, n2 = data[names[i]], data[names[j]]
            dist_m = osrm_distances[i][j]
            dist_matrix[i][j] = int(dist_m)
            
            time_matrix[i][j] = int(osrm_durations[i][j] / 60)
            
            delta_h = n2['alt'] - n1['alt']
            dist_km = dist_m / 1000.0
            payload_malus = 40 * (n1['demand'] / van_capacity)
            
            if delta_h > 0:
                energy_arc = (base_consumption + payload_malus) * dist_km + (22 * delta_h)
            else:
                energy_arc = (base_consumption + payload_malus) * dist_km + (11 * delta_h)
                
            min_possible_energy = 50 * dist_km
            energy_matrix[i][j] = max(int(energy_arc), int(min_possible_energy))
            
    return dist_matrix, time_matrix, energy_matrix

dist_M, time_M, energy_M = build_matrices_with_osrm(nodes_data)

# --- 4. OR-TOOLS COGNITIVE ENGINE ---
def solve_vrp(max_vehicles):
    hub_index = 0
    for idx, name in enumerate(node_names):
        if nodes_data[name]['type'] == 'hub':
            hub_index = idx
            break

    manager = pywrapcp.RoutingIndexManager(num_nodes, max_vehicles, hub_index)
    routing = pywrapcp.RoutingModel(manager)

    def cost_callback(from_index, to_index):
        n_from = manager.IndexToNode(from_index)
        n_to = manager.IndexToNode(to_index)
        cost_time = (time_M[n_from][n_to] / 60.0) * driver_rate
        cost_energy = (energy_M[n_from][n_to] / 1000.0) * kwh_cost
        return int((cost_time + cost_energy) * 100)
        
    cost_idx = routing.RegisterTransitCallback(cost_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(cost_idx)
    
    for v in range(max_vehicles):
        routing.SetFixedCostOfVehicle(fixed_cost * 100, v)

    # --- 4.1 PAYLOAD ---
    def demand_callback(from_index):
        return int(nodes_data[node_names[manager.IndexToNode(from_index)]]['demand'])
    demand_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimension(demand_idx, 0, int(van_capacity), True, 'Payload')

    # --- 4.2 TIME DIMENSION (CORRIGÉ AVEC BUFFER DE RECHARGE) ---
    def transit_time_callback(from_index, to_index):
        n_from = manager.IndexToNode(from_index)
        n_to = manager.IndexToNode(to_index)
        driving_time = int(time_M[n_from][n_to])
        
        # Temps de manutention fixe quand on quitte un client (25 min)
        if n_from != hub_index and nodes_data[node_names[n_from]]['type'] == 'client':
            driving_time += 25
            
        # ⚡ BUFFER DE SÉCURITÉ : Si la destination est une borne Neogy, 
        # on provisionne 20 minutes d'overhead logistique dans le budget du solver
        if nodes_data[node_names[n_to]]['type'] == 'charge':
            driving_time += 20
            
        return driving_time

    transit_time_idx = routing.RegisterTransitCallback(transit_time_callback)
    routing.AddDimension(transit_time_idx, 0, 300, True, 'Time')
    time_dimension = routing.GetDimensionOrDie('Time')

    # --- 4.3 EV BATTERY & PARTIAL SLACK RECOVERY ---
    battery_max_wh = int(battery_cap_kwh * 1000)

    def energy_callback(from_index, to_index):
        n_from = manager.IndexToNode(from_index)
        n_to = manager.IndexToNode(to_index)
        
        if n_to != hub_index and nodes_data[node_names[n_to]]['type'] == 'charge':
            return int(energy_M[n_from][n_to] - battery_max_wh)
        return int(energy_M[n_from][n_to])
        
    energy_idx = routing.RegisterTransitCallback(energy_callback)
    routing.AddDimension(energy_idx, battery_max_wh, battery_max_wh, True, 'Battery')
    battery_dimension = routing.GetDimensionOrDie('Battery')

    # --- 4.4 NODE CONSTRAINTS GRAPH ---
    for i in range(num_nodes):
        index = manager.NodeToIndex(i)
        node_type = nodes_data[node_names[i]]['type']
        
        battery_dimension.SlackVar(index).SetRange(0, battery_max_wh)
        
        if i == hub_index: continue
        
        if node_type == 'charge':
            routing.AddDisjunction([index], 0)
            time_dimension.SlackVar(index).SetRange(0, 300) 
            routing.AddVariableMinimizedByFinalizer(time_dimension.SlackVar(index))
        else:
            routing.AddDisjunction([index], 1000000)
            battery_dimension.CumulVar(index).SetMax(battery_max_wh)

    # --- 4.5 HEURISTICS RUNTIME & RESOLUTION ---
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_params.time_limit.seconds = 20

    solution = routing.SolveWithParameters(search_params)
    
    status_mapping = {
        0: "ROUTING_NOT_SOLVED",
        1: "ROUTING_SUCCESS (Optimal Setup Found)",
        2: "ROUTING_FAIL (No Possible Config)",
        3: "ROUTING_FAIL_TIMEOUT (Calculations cut off by safety timer ⏱️)",
        4: "ROUTING_INVALID"
    }
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🖥️ Core Solver Status")
    st.sidebar.info(f"Status: {status_mapping.get(routing.status(), 'Unknown')}")
    
    routes = {}
    recharges = {}
    
    if solution:
        for v in range(max_vehicles):
            index = routing.Start(v)
            routes[v] = []
            recharges[v] = {}
            while not routing.IsEnd(index):
                node_idx = manager.IndexToNode(index)
                routes[v].append(node_idx)
                index = solution.Value(routing.NextVar(index))
            routes[v].append(manager.IndexToNode(index))
            
    return routes, recharges, manager, solution, battery_dimension, routing

# Execution globale et blocage de la mémoire C++
routes_dict, recharges_dict, manager, solution, battery_dimension, routing = solve_vrp(max_fleet_size)

# --- 5. POST-PROCESSING STATISTICAL ANALYSIS ---
if routes_dict:
    total_dist_m = 0
    total_time_min = 0
    total_energy_driving_wh = 0
    total_energy_charged_wh = 0
    total_linen_delivered = 0
    active_vans_count = 0
    unique_clients_visited = set()
    
    for v_id, path in routes_dict.items():
        if len(path) <= 2: continue
        active_vans_count += 1

        for index in range(len(path)-1):
            u, v_node = path[index], path[index+1]
            total_dist_m += dist_M[u][v_node]
            
            service = 25 if nodes_data[node_names[u]]['type'] == 'client' else 0
            
            charge_time = 0
            if nodes_data[node_names[v_node]]['type'] == 'charge':
                idx_u = manager.NodeToIndex(u)
                idx_v = manager.NodeToIndex(v_node)
                defect_before = solution.Value(battery_dimension.CumulVar(idx_u))
                defect_after = solution.Value(battery_dimension.CumulVar(idx_v))
                
                wh_charged = defect_before + energy_M[u][v_node] - defect_after
                wh_charged = max(0, wh_charged)
                
                total_energy_charged_wh += wh_charged
                charge_time = wh_charged * 0.0012  
                recharges_dict[v_id][v_node] = wh_charged 

            total_time_min += time_M[u][v_node] + service + charge_time
            
            if nodes_data[node_names[u]]['type'] == 'client':
                unique_clients_visited.add(node_names[u])
                total_linen_delivered += nodes_data[node_names[u]]['demand']
            total_energy_driving_wh += energy_M[u][v_node]

    driver_total_cost = (total_time_min / 60.0) * driver_rate
    energy_total_cost = (total_energy_driving_wh / 1000.0) * kwh_cost
    fleet_fixed_cost = active_vans_count * fixed_cost
    grand_total_cost = driver_total_cost + energy_total_cost + fleet_fixed_cost
    
    total_clients_in_db = sum(1 for n in nodes_data.values() if n['type'] == 'client')

    # --- 6. USER INTERFACE METRICS PRESENTATION ---
    st.markdown("---")
    st.subheader("📊 Executive Dashboard & Operational Metrics")
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        if len(unique_clients_visited) == total_clients_in_db:
            st.metric("✅ Client Coverage", f"{len(unique_clients_visited)} / {total_clients_in_db}", "100% Fulfilled")
        else:
            st.metric("⚠️ Client Coverage", f"{len(unique_clients_visited)} / {total_clients_in_db}", f"{total_clients_in_db - len(unique_clients_visited)} Unserved", delta_color="inverse")
    with kpi2:
        st.metric("💰 Total Operational Cost", f"{grand_total_cost:.2f} €")
    with kpi3:
        st.metric("🚛 Dispatched Fleet", f"{active_vans_count} Active Vans", f"Out of {max_fleet_size} available")
    with kpi4:
        st.metric("📦 Total Linen Volume", f"{int(total_linen_delivered)} kg")

    if len(unique_clients_visited) < total_clients_in_db:
        all_clients = set(name for name, d in nodes_data.items() if d['type'] == 'client')
        missing_clients = all_clients - unique_clients_visited
        st.error(f"🚨 Logistics Failure: The following {len(missing_clients)} properties could not be served within the strict 5-hour morning window:")
        st.write(", ".join(list(missing_clients)))

    with st.expander("🔍 Financial & Analytical Breakdown"):
        c_an1, c_an2, c_an3, c_an4 = st.columns(4)
        c_an1.metric("🧑‍✈️ Driver Wages", f"{driver_total_cost:.2f} €", f"{total_time_min/60:.1f} Total Hours")
        c_an2.metric("🔌 Charging Cost", f"{energy_total_cost:.2f} €", f"{total_energy_driving_wh/1000:.1f} kWh Consumed")
        c_an3.metric("🏢 Fleet Depreciation", f"{fleet_fixed_cost:.2f} €", f"{fixed_cost} €/van/day")
        c_an4.metric("🔋 Smart Recharges", f"{total_energy_charged_wh / 1000:.1f} kWh", "Partial Injections")
        st.info(f"🛣️ **Cumulative Fleet Distance Across Trentino Road Network:** {total_dist_m / 1000:.1f} km")

    col1, col2 = st.columns([2, 3])
    with col1:
        st.subheader("🚛 Route Sheet Breakdown per Vehicle")
        van_index = 0
        for v_id, path in routes_dict.items():
            if len(path) <= 2: continue
            van_index += 1
            
            v_dist = 0
            v_time = 0
            v_energy = 0
            v_steps = []
            
            for index in range(len(path)-1):
                u, v_node = path[index], path[index+1]
                v_dist += dist_M[u][v_node]
                
                service = 25 if nodes_data[node_names[u]]['type'] == 'client' else 0
                
                charge_time = 0
                u_name = node_names[u]
                if nodes_data[u_name]['type'] == 'charge':
                    wh_charged = recharges_dict[v_id].get(u, 0)
                    charge_time = wh_charged * 0.0012
                    if wh_charged > 10:  
                        u_name += f" 🔌 (+{wh_charged/1000:.1f} kWh)"
                
                v_time += time_M[u][v_node] + service + charge_time
                v_energy += energy_M[u][v_node]
                v_steps.append(u_name)
                
            v_steps.append(node_names[path[-1]])
            
            st.markdown(f"### 📦 Opel Vivaro-e #{van_index}")
            st.caption(f"**Manifest:** {' ➔ '.join(v_steps)}")
            
            sm1, sm2, sm3 = st.columns(3)
            sm1.markdown(f"📏 **{v_dist/1000:.1f} km**")
            sm2.markdown(f"⏱️ **{v_time/60:.1f} hrs** (incl. variables)")
            sm3.markdown(f"🔋 **{v_energy/1000:.1f} kWh**")
            st.markdown("---")

    with col2:
        st.subheader("🗺️ Dynamic OSRM GIS Mapping")
        m = folium.Map(location=[46.18, 11.20], zoom_start=9, control_scale=True)
        palette = ['blue', 'green', 'purple', 'orange', 'red', 'darkblue', 'darkred', 'cadetblue']
        
        for name, data in nodes_data.items():
            if data['type'] == 'hub':
                folium.Marker([data['lat'], data['lon']], tooltip=f"Central Hub: {name}", icon=folium.Icon(color='black', icon='home')).add_to(m)
            elif data['type'] == 'charge':
                folium.Marker([data['lat'], data['lon']], tooltip=f"Neogy Station: {name}", icon=folium.Icon(color='green', icon='bolt', prefix='fa')).add_to(m)
            else:
                folium.Marker([data['lat'], data['lon']], tooltip=f"{name} ({int(data['demand'])} kg)", icon=folium.Icon(color='lightgray', icon='hotel', prefix='fa')).add_to(m)

        idx = 0
        for v_id, path in routes_dict.items():
            if len(path) <= 2: continue
            route_nodes = [nodes_data[node_names[n]] for n in path]
            coords_str = ";".join([f"{node['lon']},{node['lat']}" for node in route_nodes])
            route_url = f"http://router.project-osrm.org/route/v1/driving/{coords_str}?overview=full&geometries=geojson"
            
            try:
                route_response = requests.get(route_url).json()
                if "routes" in route_response and len(route_response["routes"]) > 0:
                    geojson_coords = route_response["routes"][0]["geometry"]["coordinates"]
                    folium_coords = [[coord[1], coord[0]] for coord in geojson_coords]
                    folium.PolyLine(folium_coords, color=palette[idx % len(palette)], weight=5, opacity=0.85, tooltip=f"Van {idx + 1}").add_to(m)
            except:
                fallback_coords = [[node['lat'], node['lon']] for node in route_nodes]
                folium.PolyLine(fallback_coords, color=palette[idx % len(palette)], weight=3, opacity=0.7).add_to(m)
            idx += 1
                
        st_folium(m, width=700, height=500, returned_objects=[])
else:
    st.error("❌ The mathematical engine could not resolve a valid configuration. Please check your structural vehicle capacity constraints or battery levels.")