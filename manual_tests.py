from project_starter import MultiAgentWorkflow
import logging
from datetime import datetime

# Constants for dates
REQUEST_DATE = "2025-04-01"
DELIVERY_DATE = "2025-04-15"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("workflow.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

def process_request(request_text: str) -> str:
    """Process a request using the MultiAgentWorkflow."""
    try:
        workflow = MultiAgentWorkflow()
        return workflow.run(request_text)
    except Exception as e:
        logger.error(f"An error occurred while processing the request: {e}")
        raise

# Examples of common requests
sample_request1 = """
    Could you verify if the following office materials are in stock for the forthcoming event?
    - 3500 units of letter-sized premium paper
    - 2000 units of tabloid-sized standard paper
    - 500 units of heavyweight paper in various hues
    
    Kindly confirm availability and specify the anticipated shipment date.
    """

sample_request2 = f"""
    I need to order the following office materials for the event: 
    - 500 units of letter-sized premium paper
    - 2000 units of 
    Please ensure delivery by {DELIVERY_DATE}. Thank you. (Order date: {REQUEST_DATE})
    """

# Process the 1st request and log the response
try:
    response = process_request(sample_request1)
    print(response)
except Exception as e:
    logger.error(f"Failed to process sample_request1: {e}")

# Process the 2nd request and log the response
try:
    response = process_request(sample_request2)
    print(response)
except Exception as e:
    logger.error(f"Failed to process sample_request2: {e}")