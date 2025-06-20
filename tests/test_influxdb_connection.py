from influxdb_client import InfluxDBClient
import traceback
from config.influxdb_config import get_influxdb_config

def test_connection():
    """Test the connection to InfluxDB using JSON configuration"""
    print("Testing InfluxDB connection...")
    
    try:
        # Get configuration from JSON file
        url, token, org, bucket = get_influxdb_config()
        
        # Print configuration (without full token for security)
        print(f"URL: {url}")
        print(f"ORG: {org}")
        print(f"BUCKET: {bucket}")
        print(f"TOKEN: {'*' * 10}...{token[-5:] if token else 'Not set'}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return False
    
    try:
        # Create InfluxDB client
        print("\nConnecting to InfluxDB...")
        client = InfluxDBClient(url=url, token=token, org=org)
        
        # Test a simple query to check if the connection works
        # This will list the available buckets
        buckets_api = client.buckets_api()
        buckets = buckets_api.find_buckets().buckets
        
        print("\nConnection successful!")
        print(f"Available buckets: {[bucket.name for bucket in buckets]}")
        
        # Check if our bucket exists
        our_bucket = bucket
        if our_bucket in [b.name for b in buckets]:
            print(f"✅ Bucket '{our_bucket}' exists!")
        else:
            print(f"❌ Bucket '{our_bucket}' not found. You may need to create it.")
        
        # List organizations
        print("\nListing organizations:")
        orgs_api = client.organizations_api()
        orgs = orgs_api.find_organizations()
        
        print(f"Available organizations: {[org.name for org in orgs]}")
        print(f"Organization IDs: {[org.id for org in orgs]}")
        
        # Close the client
        client.close()
        
        return True
    except Exception as e:
        print(f"\nConnection failed: {str(e)}")
        print("\nDetailed error information:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_connection() 