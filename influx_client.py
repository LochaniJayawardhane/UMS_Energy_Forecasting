from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS
import pandas as pd
import os
import json
from datetime import datetime
import pathlib

class InfluxClient:
    """
    Client for interacting with InfluxDB to retrieve meter data
    """
    def __init__(self):
        # Load configuration from JSON file
        self._load_config()
        
        # Initialize InfluxDB client
        self.client = InfluxDBClient(
            url=self.url,
            token=self.token,
            org=self.org
        )
        
        # Initialize query API
        self.query_api = self.client.query_api()
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        
        print(f"InfluxDB client initialized: URL={self.url}, org={self.org}")
    
    def _load_config(self):
        """Load configuration from JSON file"""
        from config.influxdb_config import load_influxdb_config
        
        try:
            config = load_influxdb_config()
            self.url = config['url']
            self.token = config['token']
            self.org = config['org']
            self.bucket = config['bucket']
        except Exception as e:
            raise ValueError(f"Error loading InfluxDB configuration: {str(e)}")
    
    def get_meter_data(self, meter_id, meter_type, start_date=None, end_date=None, limit=None):
        """
        Get data for a specific meter from InfluxDB.
        If start_date and end_date are not provided, fetches all available data.
        
        Args:
            meter_id (str): ID of the meter
            meter_type (str): Type of meter (electricity/water)
            start_date (str, optional): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format
            limit (int, optional): Maximum number of records to return
            
        Returns:
            DataFrame with DateTime, Consumption, and Temperature columns
        """
        # Validate meter type
        if meter_type not in ["electricity", "water"]:
            raise ValueError("meter_type must be 'electricity' or 'water'")
        
        # Use the correct measurement name from the InfluxDB structure
        measurement = "energy_consumption"
        
        # Build Flux query
        query = f'''
        from(bucket: "{self.bucket}")
          |> range('''
        
        # Add date range if provided
        if start_date and end_date:
            query += f'start: {start_date}T00:00:00Z, stop: {end_date}T23:59:59Z'
        else:
            query += 'start: 0, stop: now()'
            
        query += f''')
          |> filter(fn: (r) => r._measurement == "{measurement}")
          |> filter(fn: (r) => r.meter_type == "{meter_type}")
          |> filter(fn: (r) => r.meter_id == "{meter_id}")
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        
        # Add limit if provided
        if limit is not None and limit > 0:
            query += f'\n  |> limit(n: {limit})'
        
        print(f"Executing query for {meter_type} meter {meter_id}")
        print(f"Date range: {start_date or 'all'} to {end_date or 'now'}")
        if limit:
            print(f"Limit: {limit} records")
        
        try:
            # Execute query
            result = self.query_api.query_data_frame(query)
            
            if isinstance(result, list):
                if not result:  # Empty list
                    return pd.DataFrame(columns=["DateTime", "Consumption", "Temperature"])
                result = pd.concat(result)
            
            # Check if we got any data
            if result.empty:
                print(f"No data found for {meter_type} meter {meter_id}")
                return pd.DataFrame(columns=["DateTime", "Consumption", "Temperature"])
            
            # Process result
            df = result[["_time", "consumption", "temperature"]].copy()
            df.columns = ["DateTime", "Consumption", "Temperature"]
            
            # Convert DateTime to pandas datetime
            df["DateTime"] = pd.to_datetime(df["DateTime"])
            
            # Sort by DateTime
            df = df.sort_values("DateTime")
            
            print(f"Retrieved {len(df)} records from InfluxDB")
            return df
            
        except Exception as e:
            print(f"Error querying InfluxDB: {str(e)}")
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=["DateTime", "Consumption", "Temperature"])
    
    def get_last_record(self, meter_id, meter_type):
        """
        Get the most recent record for a specific meter from InfluxDB.
        
        Args:
            meter_id (str): ID of the meter
            meter_type (str): Type of meter (electricity/water)
            
        Returns:
            DataFrame with a single row containing the most recent record
        """
        # Validate meter type
        if meter_type not in ["electricity", "water"]:
            raise ValueError("meter_type must be 'electricity' or 'water'")
        
        # Use the correct measurement name from the InfluxDB structure
        measurement = "energy_consumption"
        
        # Build Flux query to get the most recent record
        query = f'''
        from(bucket: "{self.bucket}")
          |> range(start: 0, stop: now())
          |> filter(fn: (r) => r._measurement == "{measurement}")
          |> filter(fn: (r) => r.meter_type == "{meter_type}")
          |> filter(fn: (r) => r.meter_id == "{meter_id}")
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> sort(columns: ["_time"], desc: true)
          |> limit(n: 1)
        '''
        
        print(f"Fetching most recent record for {meter_type} meter {meter_id}")
        
        try:
            # Execute query
            result = self.query_api.query_data_frame(query)
            
            if isinstance(result, list):
                if not result:  # Empty list
                    return None
                result = pd.concat(result)
            
            # Check if we got any data
            if result.empty:
                print(f"No data found for {meter_type} meter {meter_id}")
                return None
            
            # Process result
            df = result[["_time", "consumption", "temperature"]].copy()
            df.columns = ["DateTime", "Consumption", "Temperature"]
            
            # Convert DateTime to pandas datetime
            df["DateTime"] = pd.to_datetime(df["DateTime"])
            
            print(f"Retrieved last record from {df['DateTime'].iloc[0]}")
            return df
            
        except Exception as e:
            print(f"Error querying InfluxDB: {str(e)}")
            return None
            
    def close(self):
        """Close the InfluxDB client connection"""
        if hasattr(self, 'client'):
            self.client.close()
            print("InfluxDB client connection closed") 