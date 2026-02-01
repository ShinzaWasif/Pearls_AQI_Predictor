import os
from google.cloud import aiplatform
from google.oauth2 import service_account
# We use the preview module for the BigQuery-backed Feature Store
from vertexai.resources.preview.feature_store import FeatureGroup, utils as fs_utils
from dotenv import load_dotenv

load_dotenv()

def setup_vertex_feature_store():
    # 1. Load Credentials
    service_account_path = os.getenv("GCP_SERVICE_ACCOUNT_JSON")
    credentials = service_account.Credentials.from_service_account_file(service_account_path)

    # 2. Initialize
    aiplatform.init(
        project=os.getenv("GCP_PROJECT_ID"),
        location="us-central1",
        credentials=credentials
    )

    print("🚀 Initializing Vertex AI Feature Store setup...")

    # 3. Define the BigQuery Source
    bq_uri = f"bq://{os.getenv('GCP_PROJECT_ID')}.{os.getenv('BQ_DATASET_ID')}.{os.getenv('BQ_TABLE_ID')}"
    
    # We use fs_utils which we imported from the preview module
    bq_source = fs_utils.FeatureGroupBigQuerySource(
        uri=bq_uri,
        entity_id_columns=["city"] 
    )

    try:
        # 4. Create or Get the Feature Group
        karachi_fg = FeatureGroup.create(
            name="karachi_aqi_feature_group",
            source=bq_source,
            description="Feature Store for Karachi Air Quality Prediction"
        )
        print(f"✅ Feature Group Created: {karachi_fg.resource_name}")
    except Exception as e:
        # If it already exists, we just print a message
        if "already exists" in str(e).lower():
            print("ℹ️ Feature Group 'karachi_aqi_feature_group' already exists. Skipping creation.")
        else:
            print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    setup_vertex_feature_store()