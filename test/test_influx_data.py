#!/usr/bin/env python
import sys
import pandas as pd
from influx_client import InfluxClient

def test_influx_data_retrieval(meter_id, meter_type, limit=20):
    """
    Test script to retrieve a limited number of rows from InfluxDB for a specific meter.
    
    Args:
        meter_id (str): ID of the meter to retrieve data for
        meter_type (str): Type of meter ('electricity' or 'water')
        limit (int): Maximum number of rows to retrieve (default: 20)
    """
    print(f"\nTesting data retrieval for {meter_type} meter {meter_id}...")
    print(f"Attempting to retrieve up to {limit} rows from InfluxDB")
    
    try:
        # Initialize InfluxDB client
        influx_client = InfluxClient()
        
        # Retrieve data with limit
        data = influx_client.get_meter_data(
            meter_id=meter_id,
            meter_type=meter_type,
            limit=limit
        )
        
        # Close the client connection
        influx_client.close()
        
        # Display results
        if data.empty:
            print(f"\n❌ No data found for {meter_type} meter {meter_id}")
            return False
        
        print(f"\n✅ Successfully retrieved {len(data)} rows for {meter_type} meter {meter_id}")
        
        # Display data summary
        print("\nData Summary:")
        print(f"- Date Range: {data['DateTime'].min()} to {data['DateTime'].max()}")
        print(f"- Consumption: min={data['Consumption'].min():.2f}, max={data['Consumption'].max():.2f}, avg={data['Consumption'].mean():.2f}")
        print(f"- Temperature: min={data['Temperature'].min():.2f}, max={data['Temperature'].max():.2f}, avg={data['Temperature'].mean():.2f}")
        
        # Display first 5 rows
        print("\nFirst 5 rows:")
        print(data.head(5).to_string())
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error retrieving data: {str(e)}")
        return False

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) < 3:
        print("Usage: python test_influx_data.py <meter_id> <meter_type> [limit]")
        print("Example: python test_influx_data.py test_001 electricity 20")
        sys.exit(1)
    
    # Parse arguments
    meter_id = sys.argv[1]
    meter_type = sys.argv[2].lower()
    
    # Validate meter type
    if meter_type not in ["electricity", "water"]:
        print("Error: meter_type must be 'electricity' or 'water'")
        sys.exit(1)
    
    # Parse optional limit
    limit = 20
    if len(sys.argv) > 3:
        try:
            limit = int(sys.argv[3])
        except ValueError:
            print("Error: limit must be an integer")
            sys.exit(1)
    
    # Run the test
    success = test_influx_data_retrieval(meter_id, meter_type, limit)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1) 