import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import heapq
import random
from datetime import datetime, timedelta

class SmartIrrigationSystem:
    def __init__(self, num_zones=6):
        """Initialize the Smart Irrigation System with a specified number of zones"""
        self.num_zones = num_zones
        self.zones = {}
        self.graph = {}
        self.reservoir_level = 100  # Initial reservoir level (0-100)
        self.reservoir_capacity = 100
        self.irrigation_schedule = []
        self.current_time = datetime.now()
        
        # Generate zones with initial moisture levels and positions
        self.generate_zones()
        
        # Generate connections between zones (graph edges)
        self.generate_graph()
    
    def generate_zones(self):
        """Generate irrigation zones with random positions and moisture levels"""
        # Reservoir is zone 0
        self.zones[0] = {
            'name': 'Reservoir',
            'moisture': 100,  # Reservoir is always at 100% moisture
            'position': (0, 0),
            'water_requirement': 0,  # Reservoir doesn't need water
            'soil_type': 'Water',
            'crop_type': None,
            'area': 0
        }
        
        # Create other zones
        zone_types = ['Vegetables', 'Fruits', 'Flowers', 'Lawn', 'Trees']
        soil_types = ['Clay', 'Sandy', 'Loam', 'Silt', 'Peat']
        
        for i in range(1, self.num_zones + 1):
            # Random position (x, y) where x and y are between -5 and 5
            position = (random.uniform(-5, 5), random.uniform(-5, 5))
            
            # Random initial moisture level between 20% and 80%
            moisture = random.uniform(20, 80)
            
            # Random area between 50 and 200 square meters
            area = random.uniform(50, 200)
            
            # Random soil and crop types
            soil_type = random.choice(soil_types)
            crop_type = random.choice(zone_types)
            
            # Calculate water requirement based on moisture level and area
            # The lower the moisture, the higher the requirement
            water_requirement = (100 - moisture) * area / 100
            
            self.zones[i] = {
                'name': f'Zone {i}',
                'moisture': moisture,
                'position': position,
                'water_requirement': water_requirement,
                'soil_type': soil_type,
                'crop_type': crop_type,
                'area': area
            }
    
    def generate_graph(self):
        """Generate the graph representing connections between zones"""
        # Initialize graph
        for i in range(self.num_zones + 1):
            self.graph[i] = []
        
        # Connect reservoir to all zones
        for i in range(1, self.num_zones + 1):
            distance = self.calculate_distance(self.zones[0]['position'], self.zones[i]['position'])
            water_loss = distance * 0.05  # Water loss is proportional to distance
            energy_cost = distance * 0.1  # Energy cost is proportional to distance
            
            # Weight is a combination of distance, water loss, and energy cost
            weight = distance + water_loss + energy_cost
            
            # Add edge from reservoir to zone
            self.graph[0].append((i, weight))
            
            # Add edge from zone to reservoir (bidirectional)
            self.graph[i].append((0, weight))
        
        # Connect zones to each other if they are close enough
        for i in range(1, self.num_zones + 1):
            for j in range(i + 1, self.num_zones + 1):
                distance = self.calculate_distance(self.zones[i]['position'], self.zones[j]['position'])
                
                # Connect zones if they are close enough (distance < 7)
                if distance < 7:
                    water_loss = distance * 0.05
                    energy_cost = distance * 0.1
                    weight = distance + water_loss + energy_cost
                    
                    self.graph[i].append((j, weight))
                    self.graph[j].append((i, weight))
    
    def calculate_distance(self, pos1, pos2):
        """Calculate Euclidean distance between two positions"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def dijkstra(self, start_node):
        """Implement Dijkstra's algorithm to find shortest paths from start_node to all other nodes"""
        # Priority queue for nodes to visit
        queue = [(0, start_node)]
        
        # Distance from start_node to each node (initialized to infinity)
        distances = {node: float('infinity') for node in self.graph}
        distances[start_node] = 0
        
        # Previous node in optimal path
        previous = {node: None for node in self.graph}
        
        while queue:
            # Get node with smallest distance
            current_distance, current_node = heapq.heappop(queue)
            
            # If we've already found a shorter path to the current node, ignore it
            if current_distance > distances[current_node]:
                continue
            
            # Check all neighbors of current node
            for neighbor, weight in self.graph[current_node]:
                distance = current_distance + weight
                
                # If we found a shorter path to neighbor, update it
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current_node
                    heapq.heappush(queue, (distance, neighbor))
        
        return distances, previous
    
    def get_shortest_path(self, start_node, end_node):
        """Get the shortest path from start_node to end_node"""
        _, previous = self.dijkstra(start_node)
        
        path = []
        current = end_node
        
        # If there's no path to end_node, return empty path
        if previous[end_node] is None and start_node != end_node:
            return path
        
        # Reconstruct the path
        while current is not None:
            path.append(current)
            current = previous[current]
        
        # Reverse the path to get start_node -> end_node
        return path[::-1]
    
    def update_moisture_levels(self):
        """Simulate natural moisture loss over time"""
        for i in range(1, self.num_zones + 1):
            # Moisture decreases by 0.5-2% randomly
            moisture_loss = random.uniform(0.5, 2)
            self.zones[i]['moisture'] = max(0, self.zones[i]['moisture'] - moisture_loss)
            
            # Update water requirement based on new moisture level
            self.zones[i]['water_requirement'] = (100 - self.zones[i]['moisture']) * self.zones[i]['area'] / 100
    
    def irrigate_zone(self, zone_id, amount):
        """Irrigate a specific zone with a given amount of water"""
        if self.reservoir_level >= amount:
            # Update reservoir level
            self.reservoir_level -= amount
            
            # Calculate moisture increase
            moisture_increase = amount * 100 / self.zones[zone_id]['area']
            
            # Update zone moisture (cap at 100%)
            self.zones[zone_id]['moisture'] = min(100, self.zones[zone_id]['moisture'] + moisture_increase)
            
            # Update water requirement
            self.zones[zone_id]['water_requirement'] = (100 - self.zones[zone_id]['moisture']) * self.zones[zone_id]['area'] / 100
            
            return True
        else:
            st.warning(f"Not enough water in reservoir to irrigate Zone {zone_id}")
            return False
    
    def optimize_irrigation(self):
        """Optimize irrigation based on moisture levels and shortest paths"""
        # Sort zones by water requirement (highest first)
        zones_to_irrigate = sorted(
            [(i, self.zones[i]['water_requirement']) for i in range(1, self.num_zones + 1)],
            key=lambda x: x[1],
            reverse=True
        )
        
        schedule = []
        
        for zone_id, requirement in zones_to_irrigate:
            # Skip zones that don't need water
            if requirement <= 0 or self.zones[zone_id]['moisture'] >= 70:
                continue
            
            # Get the shortest path from reservoir to this zone
            path = self.get_shortest_path(0, zone_id)
            
            # Skip if no path is found
            if not path:
                continue
            
            # Calculate water amount (enough to reach 85% moisture)
            target_moisture = 85
            current_moisture = self.zones[zone_id]['moisture']
            area = self.zones[zone_id]['area']
            
            water_amount = (target_moisture - current_moisture) * area / 100
            
            # Cap water amount based on reservoir level
            water_amount = min(water_amount, self.reservoir_level)
            
            # Schedule irrigation
            schedule.append({
                'zone_id': zone_id,
                'path': path,
                'water_amount': water_amount,
                'time': self.current_time + timedelta(minutes=len(schedule)*30)  # Schedule every 30 minutes
            })
            
            # Simulate irrigation
            self.irrigate_zone(zone_id, water_amount)
        
        self.irrigation_schedule = schedule
        return schedule
    
    def refill_reservoir(self, amount=None):
        """Refill the reservoir with a specified amount or to full capacity"""
        if amount is None:
            amount = self.reservoir_capacity - self.reservoir_level
        
        self.reservoir_level = min(self.reservoir_capacity, self.reservoir_level + amount)
        st.success(f"Reservoir refilled. Current level: {self.reservoir_level:.2f} units")
    
    def simulate_day(self, days=1):
        """Simulate irrigation system for a specified number of days"""
        results = []
        
        for day in range(days):
            daily_data = {
                'day': day + 1,
                'moisture_levels': {},
                'irrigation_events': []
            }
            
            # 3 irrigation cycles per day
            for cycle in range(3):
                # Update moisture levels (simulate natural loss)
                self.update_moisture_levels()
                
                # Optimize irrigation
                schedule = self.optimize_irrigation()
                
                # Record moisture levels after irrigation
                for i in range(1, self.num_zones + 1):
                    if i not in daily_data['moisture_levels']:
                        daily_data['moisture_levels'][i] = []
                    
                    daily_data['moisture_levels'][i].append(self.zones[i]['moisture'])
                
                # Record irrigation events
                for event in schedule:
                    daily_data['irrigation_events'].append({
                        'zone_id': event['zone_id'],
                        'water_amount': event['water_amount'],
                        'time': event['time']
                    })
                
                # Advance time by 8 hours
                self.current_time += timedelta(hours=8)
            
            # Refill reservoir at the end of the day
            self.refill_reservoir()
            
            results.append(daily_data)
        
        return results

    def visualize_zones(self):
        """Visualize the irrigation zones and their connections"""
        plt.figure(figsize=(12, 10))
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes with positions
        pos = {}
        node_colors = []
        node_sizes = []
        labels = {}
        
        for i in range(self.num_zones + 1):
            G.add_node(i)
            pos[i] = self.zones[i]['position']
            labels[i] = self.zones[i]['name']
            
            if i == 0:  # Reservoir
                node_colors.append('blue')
                node_sizes.append(500)
            else:
                # Color based on moisture level (red=dry, green=wet)
                moisture = self.zones[i]['moisture']
                if moisture < 30:
                    color = 'red'
                elif moisture < 60:
                    color = 'orange'
                else:
                    color = 'green'
                
                node_colors.append(color)
                node_sizes.append(300)
        
        # Add edges
        for i in self.graph:
            for j, weight in self.graph[i]:
                G.add_edge(i, j, weight=weight)
        
        # Draw the graph
        nx.draw(G, pos, with_labels=False, node_color=node_colors, node_size=node_sizes, 
                font_weight='bold', arrowsize=15, alpha=0.7, width=1.5)
        
        # Add labels with offset to avoid overlap with nodes
        label_pos = {k: (v[0], v[1] - 0.3) for k, v in pos.items()}
        nx.draw_networkx_labels(G, label_pos, labels=labels)
        
        # Add moisture level labels for zones
        for i in range(1, self.num_zones + 1):
            plt.text(pos[i][0], pos[i][1] + 0.2, 
                     f"M:{self.zones[i]['moisture']:.1f}%", 
                     horizontalalignment='center')
        
        plt.title("Smart Irrigation System - Zone Map")
        plt.axis('off')
        plt.tight_layout()
        return plt

    def visualize_moisture_over_time(self, simulation_results):
        """Visualize moisture levels over time for each zone"""
        plt.figure(figsize=(12, 6))
        
        for day in simulation_results:
            day_num = day['day']
            for zone_id, moisture_levels in day['moisture_levels'].items():
                # Convert cycle index to time of day
                x = [(day_num - 1) * 3 + cycle/3 for cycle in range(len(moisture_levels))]
                plt.plot(x, moisture_levels, 'o-', label=f"Zone {zone_id}" if day_num == 1 else "")
        
        plt.xlabel('Time (days)')
        plt.ylabel('Moisture Level (%)')
        plt.title('Moisture Levels Over Time')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        return plt

    def visualize_irrigation_path(self, path):
        """Visualize a specific irrigation path"""
        plt.figure(figsize=(12, 10))
        
        # Create graph
        G = nx.DiGraph()
        
        # Add nodes with positions
        pos = {}
        node_colors = []
        node_sizes = []
        labels = {}
        
        for i in range(self.num_zones + 1):
            G.add_node(i)
            pos[i] = self.zones[i]['position']
            labels[i] = self.zones[i]['name']
            
            if i in path:
                if i == 0:  # Reservoir
                    node_colors.append('blue')
                else:
                    node_colors.append('lightblue')
                node_sizes.append(500)
            else:
                node_colors.append('gray')
                node_sizes.append(300)
        
        # Add all edges
        for i in self.graph:
            for j, weight in self.graph[i]:
                G.add_edge(i, j, weight=weight)
        
        # Draw all edges in gray
        nx.draw_networkx_edges(G, pos, alpha=0.2, width=1.0, edge_color='gray')
        
        # Draw path edges in blue and thicker
        path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='blue', width=2.5)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.7)
        
        # Add labels
        nx.draw_networkx_labels(G, pos, labels=labels)
        
        plt.title(f"Irrigation Path to {self.zones[path[-1]]['name']}")
        plt.axis('off')
        plt.tight_layout()
        return plt

    def visualize_all_optimal_paths(self):
        """Visualize optimal irrigation paths for all zones"""
        plt.figure(figsize=(12, 10))
        
        # Create graph
        G = nx.DiGraph()
        
        # Add nodes with positions
        pos = {}
        node_colors = []
        node_sizes = []
        labels = {}
        
        for i in range(self.num_zones + 1):
            G.add_node(i)
            pos[i] = self.zones[i]['position']
            labels[i] = self.zones[i]['name']
            
            if i == 0:  # Reservoir
                node_colors.append('blue')
            else:
                # Color based on moisture level
                moisture = self.zones[i]['moisture']
                if moisture < 30:
                    color = 'red'
                elif moisture < 60:
                    color = 'orange'
                else:
                    color = 'green'
                node_colors.append(color)
            
            node_sizes.append(300)
        
        # Add all edges
        for i in self.graph:
            for j, weight in self.graph[i]:
                G.add_edge(i, j, weight=weight)
        
        # Draw all edges in light gray
        nx.draw_networkx_edges(G, pos, alpha=0.2, width=1.0, edge_color='lightgray')
        
        # Draw optimal paths
        colors = ['blue', 'purple', 'brown', 'green', 'orange', 'red']
        for idx, zone_id in enumerate(range(1, self.num_zones + 1)):
            path = self.get_shortest_path(0, zone_id)
            path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            
            color = colors[idx % len(colors)]
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color=color, width=2.0, 
                                   alpha=0.7, label=f"Path to Zone {zone_id}")
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.7)
        
        # Add labels
        nx.draw_networkx_labels(G, pos, labels=labels)
        
        plt.title("Optimal Irrigation Paths from Reservoir")
        plt.axis('off')
        plt.legend(loc='best')
        plt.tight_layout()
        return plt

    def generate_report(self, simulation_results):
        """Generate a summary report of the simulation"""
        total_water_used = 0
        irrigation_events = 0
        
        for day in simulation_results:
            for event in day['irrigation_events']:
                total_water_used += event['water_amount']
                irrigation_events += 1
        
        report = {
            'total_days': len(simulation_results),
            'total_water_used': total_water_used,
            'irrigation_events': irrigation_events,
            'average_water_per_event': total_water_used / max(1, irrigation_events),
            'final_moisture_levels': {i: self.zones[i]['moisture'] for i in range(1, self.num_zones + 1)},
            'zones_info': self.zones
        }
        
        return report

# Streamlit app
def main():
    st.title("Smart Irrigation System Dashboard")
    
    # Initialize session state
    if 'irrigation_system' not in st.session_state:
        st.session_state.irrigation_system = SmartIrrigationSystem()
    
    # Sidebar controls
    st.sidebar.header("System Controls")
    num_zones = st.sidebar.slider("Number of Zones", 1, 10, 6)
    
    if st.sidebar.button("Reset System"):
        st.session_state.irrigation_system = SmartIrrigationSystem(num_zones=num_zones)
        st.success("System reset with new configuration")
    
    # Main dashboard
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Reservoir Status")
        st.metric("Reservoir Level", f"{st.session_state.irrigation_system.reservoir_level:.1f}%")
        
        if st.button("Refill Reservoir"):
            st.session_state.irrigation_system.refill_reservoir()
    
    with col2:
        st.subheader("System Status")
        st.metric("Current Time", st.session_state.irrigation_system.current_time.strftime("%Y-%m-%d %H:%M"))
        
        if st.button("Update Moisture Levels"):
            st.session_state.irrigation_system.update_moisture_levels()
            st.success("Moisture levels updated")
    
    # Zone information
    st.subheader("Zone Information")
    zones_data = []
    for i in range(1, st.session_state.irrigation_system.num_zones + 1):
        zone = st.session_state.irrigation_system.zones[i]
        zones_data.append({
            "Zone": zone['name'],
            "Moisture": f"{zone['moisture']:.1f}%",
            "Water Requirement": f"{zone['water_requirement']:.1f}",
            "Soil Type": zone['soil_type'],
            "Crop Type": zone['crop_type'],
            "Area": f"{zone['area']:.1f} m²"
        })
    
    st.dataframe(zones_data)
    
    # Visualizations
    st.subheader("System Visualizations")
    
    # Zone map
    st.write("Zone Map")
    fig = st.session_state.irrigation_system.visualize_zones()
    st.pyplot(fig)
    
    # Optimal paths
    st.write("Optimal Irrigation Paths")
    fig = st.session_state.irrigation_system.visualize_all_optimal_paths()
    st.pyplot(fig)
    
    # Manual irrigation control
    st.subheader("Manual Irrigation")
    selected_zone = st.selectbox(
        "Select Zone",
        [f"Zone {i}" for i in range(1, st.session_state.irrigation_system.num_zones + 1)]
    )
    zone_id = int(selected_zone.split()[1])
    water_amount = st.number_input("Water Amount", min_value=0.0, max_value=100.0, value=10.0)
    
    if st.button("Irrigate Zone"):
        if st.session_state.irrigation_system.irrigate_zone(zone_id, water_amount):
            st.success(f"Zone {zone_id} irrigated with {water_amount} units of water")
            
            # Show irrigation path
            path = st.session_state.irrigation_system.get_shortest_path(0, zone_id)
            fig = st.session_state.irrigation_system.visualize_irrigation_path(path)
            st.pyplot(fig)
    
    # Simulation controls
    st.subheader("Simulation")
    simulation_days = st.number_input("Number of Days to Simulate", min_value=1, max_value=7, value=1)
    
    if st.button("Run Simulation"):
        with st.spinner("Running simulation..."):
            results = st.session_state.irrigation_system.simulate_day(days=simulation_days)
            st.success("Simulation completed!")
            
            # Display simulation results
            st.subheader("Simulation Results")
            for day in results:
                st.write(f"Day {day['day']}:")
                for event in day['irrigation_events']:
                    st.write(f"- Zone {event['zone_id']} irrigated with {event['water_amount']:.1f} units at {event['time']}")
            
            # Show moisture trends
            st.write("Moisture Trends Over Time")
            fig = st.session_state.irrigation_system.visualize_moisture_over_time(results)
            st.pyplot(fig)
            
            # Generate and display report
            report = st.session_state.irrigation_system.generate_report(results)
            st.subheader("Simulation Report")
            st.write(f"Total days simulated: {report['total_days']}")
            st.write(f"Total water used: {report['total_water_used']:.2f} units")
            st.write(f"Total irrigation events: {report['irrigation_events']}")
            st.write(f"Average water per event: {report['average_water_per_event']:.2f} units")
            
            st.write("Final moisture levels:")
            for zone_id, moisture in report['final_moisture_levels'].items():
                st.write(f"Zone {zone_id}: {moisture:.2f}%")

if __name__ == "__main__":
    main() 