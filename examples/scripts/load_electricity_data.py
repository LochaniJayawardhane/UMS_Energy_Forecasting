import pandas as pd
import os
from datetime import datetime
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import time
from config.influxdb_config import get_influxdb_config

def load_meter_data(meter_type="electricity"):
    """Load meter data from CSV file into InfluxDB
    
    Args:
        meter_type: Type of meter data (e.g., 'electricity', 'water', 'gas')
    """
    print(f"Loading {meter_type.title()} Meter Data to InfluxDB")
    print("=" * (len(meter_type) + 30))
    
    try:
        # Get InfluxDB connection parameters from JSON configuration
        url, token, org, bucket = get_influxdb_config()
    
    print(f"InfluxDB Connection:")
    print(f"URL: {url}")
    print(f"Organization: {org}")
    print(f"Bucket: {bucket}")
    print(f"Token: {'*' * 5}...{token[-5:] if token else 'Not set'}")
    except Exception as e:
        print(f"Error loading InfluxDB configuration: {e}")
        return
    
    # Define the CSV file path based on meter type
    csv_file = f"data/Merged_{meter_type.title()}_Temperature.csv"
    
    # Check if the file exists
    if not os.path.exists(csv_file):
        print(f"❌ Error: File {csv_file} not found.")
        return
    
    # Read the CSV file
    print(f"\nReading data from {csv_file}...")
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ Successfully read {len(df)} rows from CSV file.")
        
        # Display the first few rows
        print("\nSample data:")
        print(df.head())
        
        # Display data info
        print("\nData information:")
        print(f"Columns: {list(df.columns)}")
        print(f"Date range: {df['DateTime'].min()} to {df['DateTime'].max()}")
        print(f"Consumption range: {df['Consumption'].min()} to {df['Consumption'].max()}")
        print(f"Temperature range: {df['Temperature'].min()} to {df['Temperature'].max()}")
    except Exception as e:
        print(f"❌ Error reading CSV file: {str(e)}")
        return
    
    # Ask for meter ID
    meter_id = input(f"\nEnter {meter_type} meter ID (e.g., 'meter1'): ")
    if not meter_id:
        meter_id = f"{meter_type}_meter1"
        print(f"Using default meter ID: {meter_id}")
    
    # Connect to InfluxDB
    print("\nConnecting to InfluxDB...")
    try:
        client = InfluxDBClient(url=url, token=token, org=org)
        write_api = client.write_api(write_options=SYNCHRONOUS)
        
        # Convert data to InfluxDB points
        print("Converting data to InfluxDB points...")
        
        # Process in batches to avoid memory issues
        batch_size = 1000
        total_batches = (len(df) + batch_size - 1) // batch_size
        
        # Use a generic measurement name
        measurement_name = "energy_consumption"
        
        for i in range(0, len(df), batch_size):
            start_time = time.time()
            batch = df.iloc[i:i+batch_size]
            
            points = []
            for _, row in batch.iterrows():
                try:
                    # Parse datetime
                    timestamp = pd.to_datetime(row['DateTime'])
                    
                    # Create point with meter_type tag
                    point = Point(measurement_name) \
                        .tag("meter_id", meter_id) \
                        .tag("meter_type", meter_type) \
                        .field("consumption", float(row['Consumption'])) \
                        .field("temperature", float(row['Temperature'])) \
                        .time(timestamp, WritePrecision.NS)
                    
                    points.append(point)
                except (KeyError, ValueError) as e:
                    print(f"Error processing row: {e}")
                    continue
            
            # Write batch to InfluxDB
            write_api.write(bucket=bucket, record=points)
            
            batch_num = i // batch_size + 1
            elapsed = time.time() - start_time
            print(f"Batch {batch_num}/{total_batches}: Wrote {len(points)} points in {elapsed:.2f} seconds")
        
        print(f"\n✅ Successfully loaded {meter_type} meter data into InfluxDB!")
        print(f"Total points written: {len(df)}")
        print(f"Bucket: {bucket}")
        print(f"Measurement: {measurement_name}")
        print(f"Meter ID: {meter_id}")
        print(f"Meter Type: {meter_type}")
        
    except Exception as e:
        print(f"❌ Error writing to InfluxDB: {str(e)}")
    finally:
        if 'client' in locals():
            client.close()
            print("InfluxDB connection closed.")

# For backward compatibility
def load_electricity_data():
    """Legacy function to load electricity data"""
    load_meter_data(meter_type="electricity")

if __name__ == "__main__":
    # Ask for meter type
    print("Available meter types: electricity, water, gas")
    meter_type = input("Enter meter type (default: electricity): ").lower()
    if not meter_type:
        meter_type = "electricity"
    
    load_meter_data(meter_type=meter_type) 