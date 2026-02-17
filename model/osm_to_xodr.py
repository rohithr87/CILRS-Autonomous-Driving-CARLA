#!/usr/bin/env python3
"""
Convert CUSAT campus OpenStreetMap data to OpenDRIVE for CARLA
Filters roads within campus boundary and creates .xodr file
"""

import xml.etree.ElementTree as ET
import math
import sys
import os

# =================================================================
# CUSAT Campus Boundary (approximate)
# =================================================================
CUSAT_BOUNDS = {
    'min_lat': 10.0420,
    'max_lat': 10.0530,
    'min_lon': 76.3150,
    'max_lon': 76.3350,
}

# =================================================================
# Helper Functions
# =================================================================

def lat_lon_to_xy(lat, lon, ref_lat, ref_lon):
    """Convert lat/lon to local XY coordinates (meters)"""
    # Earth radius
    R = 6371000.0
    
    # Convert to radians
    lat_r = math.radians(lat)
    lon_r = math.radians(lon)
    ref_lat_r = math.radians(ref_lat)
    ref_lon_r = math.radians(ref_lon)
    
    # Simple equirectangular projection
    x = R * (lon_r - ref_lon_r) * math.cos(ref_lat_r)
    y = R * (lat_r - ref_lat_r)
    
    return x, y


def is_in_cusat(lat, lon):
    """Check if point is within CUSAT campus bounds"""
    return (CUSAT_BOUNDS['min_lat'] <= lat <= CUSAT_BOUNDS['max_lat'] and
            CUSAT_BOUNDS['min_lon'] <= lon <= CUSAT_BOUNDS['max_lon'])


def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def angle(x1, y1, x2, y2):
    return math.atan2(y2 - y1, x2 - x1)


# =================================================================
# Parse OSM File
# =================================================================

def parse_osm(osm_file):
    """Parse OSM file and extract roads within CUSAT campus"""
    print(f"Parsing {osm_file}...")
    
    tree = ET.parse(osm_file)
    root = tree.getroot()
    
    # Parse all nodes (lat/lon points)
    nodes = {}
    for node in root.findall('node'):
        node_id = node.get('id')
        lat = float(node.get('lat'))
        lon = float(node.get('lon'))
        nodes[node_id] = (lat, lon)
    
    print(f"  Total nodes: {len(nodes)}")
    
    # Road types we want
    road_types = {
        'motorway', 'trunk', 'primary', 'secondary', 'tertiary',
        'residential', 'service', 'unclassified', 'living_street',
        'motorway_link', 'trunk_link', 'primary_link', 'secondary_link',
        'tertiary_link', 'road'
    }
    
    # Parse ways (roads)
    roads = []
    for way in root.findall('way'):
        tags = {}
        for tag in way.findall('tag'):
            tags[tag.get('k')] = tag.get('v')
        
        # Only keep highways (roads)
        if 'highway' not in tags:
            continue
        
        highway_type = tags['highway']
        if highway_type not in road_types:
            continue
        
        # Get node references
        nd_refs = [nd.get('ref') for nd in way.findall('nd')]
        
        # Get coordinates
        coords = []
        in_campus = False
        for ref in nd_refs:
            if ref in nodes:
                lat, lon = nodes[ref]
                coords.append((lat, lon))
                if is_in_cusat(lat, lon):
                    in_campus = True
        
        # Only keep roads that pass through campus
        if in_campus and len(coords) >= 2:
            road_name = tags.get('name', f'Road_{way.get("id")}')
            lanes = int(tags.get('lanes', '2'))
            oneway = tags.get('oneway', 'no') == 'yes'
            
            roads.append({
                'id': way.get('id'),
                'name': road_name,
                'type': highway_type,
                'coords': coords,
                'lanes': lanes,
                'oneway': oneway,
            })
    
    print(f"  Roads in CUSAT area: {len(roads)}")
    
    # Print road names
    for road in roads:
        if road['name'] and not road['name'].startswith('Road_'):
            print(f"    📍 {road['name']} ({road['type']}, {len(road['coords'])} points)")
    
    return roads


# =================================================================
# Generate OpenDRIVE XML
# =================================================================

def generate_opendrive(roads, output_file):
    """Generate OpenDRIVE .xodr file from parsed roads"""
    
    # Reference point (center of CUSAT campus)
    ref_lat = (CUSAT_BOUNDS['min_lat'] + CUSAT_BOUNDS['max_lat']) / 2
    ref_lon = (CUSAT_BOUNDS['min_lon'] + CUSAT_BOUNDS['max_lon']) / 2
    
    print(f"\nReference point: ({ref_lat:.6f}, {ref_lon:.6f})")
    
    # Convert all roads to XY coordinates
    road_data = []
    for road in roads:
        xy_points = []
        for lat, lon in road['coords']:
            x, y = lat_lon_to_xy(lat, lon, ref_lat, ref_lon)
            xy_points.append((x, y))
        
        # Filter out very short segments
        total_length = 0
        for i in range(len(xy_points) - 1):
            total_length += distance(xy_points[i][0], xy_points[i][1],
                                     xy_points[i+1][0], xy_points[i+1][1])
        
        if total_length > 5.0:  # At least 5 meters
            road_data.append({
                'id': road['id'],
                'name': road['name'],
                'points': xy_points,
                'length': total_length,
                'lanes': road['lanes'],
                'oneway': road['oneway'],
            })
    
    print(f"Roads after filtering: {len(road_data)}")
    
    # Calculate map bounds
    all_x = [p[0] for r in road_data for p in r['points']]
    all_y = [p[1] for r in road_data for p in r['points']]
    
    if not all_x:
        print("ERROR: No road data found!")
        return False
    
    print(f"Map size: {max(all_x)-min(all_x):.0f}m x {max(all_y)-min(all_y):.0f}m")
    
    # Build OpenDRIVE XML
    xodr = build_opendrive_xml(road_data)
    
    # Write file
    with open(output_file, 'w') as f:
        f.write(xodr)
    
    print(f"\n✅ OpenDRIVE file saved: {output_file}")
    print(f"   Size: {os.path.getsize(output_file) / 1024:.1f} KB")
    
    return True


def build_opendrive_xml(road_data):
    """Build the OpenDRIVE XML string"""
    
    header = '''<?xml version="1.0" encoding="UTF-8"?>
<OpenDRIVE>
    <header revMajor="1" revMinor="4" name="CUSAT_Campus" version="1.0"
            date="2025-02-16" north="0.0" south="0.0" east="0.0" west="0.0">
        <geoReference><![CDATA[+proj=tmerc +lat_0=10.0475 +lon_0=76.3250 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs]]></geoReference>
    </header>
'''
    
    roads_xml = ""
    junction_id = 1000
    
    for idx, road in enumerate(road_data):
        road_id = idx + 1
        points = road['points']
        
        if len(points) < 2:
            continue
        
        # Calculate road geometry (line segments)
        geometry_xml = ""
        s = 0.0  # Running distance along road
        
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            seg_length = distance(x1, y1, x2, y2)
            
            if seg_length < 0.1:
                continue
            
            hdg = angle(x1, y1, x2, y2)
            
            geometry_xml += f'''
            <geometry s="{s:.4f}" x="{x1:.4f}" y="{y1:.4f}" hdg="{hdg:.6f}" length="{seg_length:.4f}">
                <line/>
            </geometry>'''
            
            s += seg_length
        
        total_length = s
        if total_length < 1.0:
            continue
        
        # Lane width
        lane_width = 3.5
        
        # Build road XML
        road_xml = f'''
    <road name="{road['name']}" length="{total_length:.4f}" id="{road_id}" junction="-1">
        <link/>
        <planView>{geometry_xml}
        </planView>
        <elevationProfile>
            <elevation s="0.0" a="0.0" b="0.0" c="0.0" d="0.0"/>
        </elevationProfile>
        <lateralProfile/>
        <lanes>
            <laneSection s="0.0">
                <left>
                    <lane id="1" type="driving" level="false">
                        <width sOffset="0.0" a="{lane_width}" b="0.0" c="0.0" d="0.0"/>
                    </lane>
                    <lane id="2" type="sidewalk" level="false">
                        <width sOffset="0.0" a="2.0" b="0.0" c="0.0" d="0.0"/>
                    </lane>
                </left>
                <center>
                    <lane id="0" type="none" level="false"/>
                </center>
                <right>
                    <lane id="-1" type="driving" level="false">
                        <width sOffset="0.0" a="{lane_width}" b="0.0" c="0.0" d="0.0"/>
                    </lane>
                    <lane id="-2" type="sidewalk" level="false">
                        <width sOffset="0.0" a="2.0" b="0.0" c="0.0" d="0.0"/>
                    </lane>
                </right>
            </laneSection>
        </lanes>
    </road>'''
        
        roads_xml += road_xml
    
    footer = '''
</OpenDRIVE>'''
    
    return header + roads_xml + footer


# =================================================================
# Main
# =================================================================

if __name__ == '__main__':
    osm_file = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
        '~/Downloads/map.osm')
    output_file = os.path.join(os.path.dirname(osm_file), 'cusat_campus.xodr')
    
    if not os.path.exists(osm_file):
        print(f"ERROR: {osm_file} not found!")
        sys.exit(1)
    
    print("=" * 60)
    print("CUSAT Campus OSM → OpenDRIVE Converter")
    print("=" * 60)
    
    roads = parse_osm(osm_file)
    
    if roads:
        success = generate_opendrive(roads, output_file)
        if success:
            print(f"\n🎉 Done! Load in CARLA with:")
            print(f"   python3 load_cusat.py")
    else:
        print("ERROR: No roads found in CUSAT area!")
        print("Check CUSAT_BOUNDS coordinates in the script")
