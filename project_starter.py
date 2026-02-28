import ast
import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, Engine
from sqlalchemy.sql import text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("workflow.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Create an SQLite database
db_engine = create_engine("sqlite:///beavers_choice_co.db")

# List containing the different kinds of papers
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper", "category": "paper", "unit_price": 0.05},
    {"item_name": "Letter-sized paper", "category": "paper", "unit_price": 0.06},
    {"item_name": "Cardstock", "category": "paper", "unit_price": 0.15},
    {"item_name": "Colored paper", "category": "paper", "unit_price": 0.10},
    {"item_name": "Glossy paper", "category": "paper", "unit_price": 0.20},
    {"item_name": "Matte paper", "category": "paper", "unit_price": 0.18},
    {"item_name": "Recycled paper", "category": "paper", "unit_price": 0.08},
    {"item_name": "Eco-friendly paper", "category": "paper", "unit_price": 0.12},
    {"item_name": "Poster paper", "category": "paper", "unit_price": 0.25},
    {"item_name": "Banner paper", "category": "paper", "unit_price": 0.30},
    {"item_name": "Kraft paper", "category": "paper", "unit_price": 0.10},
    {"item_name": "Construction paper", "category": "paper", "unit_price": 0.07},
    {"item_name": "Wrapping paper", "category": "paper", "unit_price": 0.15},
    {"item_name": "Glitter paper", "category": "paper", "unit_price": 0.22},
    {"item_name": "Decorative paper", "category": "paper", "unit_price": 0.18},
    {"item_name": "Letterhead paper", "category": "paper", "unit_price": 0.12},
    {"item_name": "Legal-size paper", "category": "paper", "unit_price": 0.08},
    {"item_name": "Crepe paper", "category": "paper", "unit_price": 0.05},
    {"item_name": "Photo paper", "category": "paper", "unit_price": 0.25},
    {"item_name": "Uncoated paper", "category": "paper", "unit_price": 0.06},
    {"item_name": "Butcher paper", "category": "paper", "unit_price": 0.10},
    {"item_name": "Heavyweight paper", "category": "paper", "unit_price": 0.20},
    {"item_name": "Standard copy paper", "category": "paper", "unit_price": 0.04},
    {"item_name": "Bright-colored paper", "category": "paper", "unit_price": 0.12},
    {"item_name": "Patterned paper", "category": "paper", "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates", "category": "product", "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups", "category": "product", "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins", "category": "product", "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups", "category": "product", "unit_price": 0.10},  # per cup
    {"item_name": "Table covers", "category": "product", "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes", "category": "product", "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes", "category": "product", "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads", "category": "product", "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards", "category": "product", "unit_price": 0.50},  # per card
    {"item_name": "Flyers", "category": "product", "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers", "category": "product", "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags", "category": "product", "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards", "category": "product", "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders", "category": "product", "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock", "category": "specialty", "unit_price": 0.50},
    {"item_name": "80 lb text paper", "category": "specialty", "unit_price": 0.40},
    {"item_name": "250 gsm cardstock", "category": "specialty", "unit_price": 0.30},
    {"item_name": "220 gsm poster paper", "category": "specialty", "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 11235) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)


def init_database(db_engine: Engine, seed: int = 137) -> Engine:
    """
    Set up the Beaver's Choice Paper Company database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],  # Quantity involved
            "price": [],  # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise


def create_transaction(
        item_name: str,
        transaction_type: str,
        quantity: int,
        price: float,
        date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (create_transaction): Creating transaction for '{item_name}' ")

    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        logger.error(f"Error creating transaction: {e}")
        raise


def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_all_inventory): Fetching inventory as of '{as_of_date}'")

    # SQL query to compute stock levels per item as of the given date
    query = """
            SELECT item_name,
                   SUM(CASE
                           WHEN transaction_type = 'stock_orders' THEN units
                           WHEN transaction_type = 'sales' THEN -units
                           ELSE 0
                       END) as stock
            FROM transactions
            WHERE item_name IS NOT NULL
              AND transaction_date <= :as_of_date
            GROUP BY item_name
            HAVING stock > 0 \
            """

    try:
        # Execute the query with the date parameter
        result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})
    except Exception as e:
        logger.error(f"Error fetching inventory: {e}")
        return {}

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))


# INFO: Switched return type to Dict[str, int] cause of pydantic_core._pydantic_core.PydanticSerializationError: Unable to serialize unknown type: <class 'pandas.core.frame.DataFrame'>
def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> dict:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        dict with attribute 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
                  SELECT item_name,
                         COALESCE(SUM(CASE
                                          WHEN transaction_type = 'stock_orders' THEN units
                                          WHEN transaction_type = 'sales' THEN -units
                                          ELSE 0
                             END), 0) AS current_stock
                  FROM transactions
                  WHERE item_name = :item_name
                    AND transaction_date <= :as_of_date \
                  """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_stock_level): Fetching stock for '{item_name}' as of '{as_of_date}'")

    try:
        df = pd.read_sql(
            stock_query,
            db_engine,
            params={"item_name": item_name, "as_of_date": as_of_date},
        )
    except Exception as e:
        logger.error(f"Error fetching stock level: {e}")
        return {}
    
    if df.empty:
        return {"item_name": item_name, "current_stock": 0}

    return df.iloc[0].to_dict()


def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")


def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_cash_balance): Calculating cash balance as of '{as_of_date}'")

    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        logger.error(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
                      SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
                      FROM transactions
                      WHERE transaction_type = 'sales'
                        AND transaction_date <= :date
                      GROUP BY item_name
                      ORDER BY total_revenue DESC
                      LIMIT 5 \
                      """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    try:
        # Execute parameterized query
        with db_engine.connect() as conn:
            result = conn.execute(text(query), params)
            return [dict(row) for row in result]
    except Exception as e:
        logger.error(f"Error searching quote history: {e}")
        return []


################################
# YOUR MULTI AGENT STARTS HERE #
################################

# Custom imports for your multi-agent system
from typing import Dict, Literal
from dotenv import load_dotenv

from pydantic import BaseModel
from pydantic_ai import Agent, UsageLimitExceeded
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.tools import Tool
from pydantic_ai.settings import ModelSettings


# Set up and load your env parameters and instantiate your model.
load_dotenv()

# Create a model instance with a custom API URL
# This could be a self-hosted OpenAI-compatible API, Azure endpoint, or proxy
# Note: OpenAIChatModel -> AsyncOpenAI -> reads OPENAI_API_KEY and OPENAI_BASE_URL from environment variables
#       OPENAI_API_MODEL is being retrieved from the environment variables, with fallback to ""
model = OpenAIChatModel(
    model_name=os.getenv("OPENAI_API_MODEL", "gpt-4o-mini")
)

# Define tools for the agents
tool_create_transaction = Tool(
    name="create_transaction",
    description="""
        Records a new transaction (stock order or sale) with specified item name, quantity, total price, and transaction date into the database's transactions table.
    """,
    function=create_transaction,
    require_parameter_descriptions=True
)

tool_get_all_inventory = Tool(
    name="get_all_inventory",
    description="""
        Retrieves a snapshot of available inventory as of a specified date by calculating net stock (sum of stock orders minus sales) for all items with positive quantities.
    """,
    function=get_all_inventory,
    require_parameter_descriptions=True
)

tool_get_stock_level = Tool(
    name="get_stock_level",
    description="""
        Retrieves the current stock level for a specified item as of a given date.
    """,
    function=get_stock_level,
    require_parameter_descriptions=True
)

tool_get_supplier_delivery_date = Tool(
    name="get_supplier_delivery_date",
    description="""
        Estimates delivery date based on order quantity and starting date, with lead times determined by order size:
        - Up to 10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - Over 1000 units: 7 days
    """,
    function=get_supplier_delivery_date,
    require_parameter_descriptions=True
)

tool_get_cash_balance = Tool(
    name="get_cash_balance",
    description="""
        Calculates cash balance as of a specified date by subtracting total stock purchase costs (from stock_orders) from total revenue (from sales) in the transactions table up to that date.
    """,
    function=get_cash_balance,
    require_parameter_descriptions=True
)

tool_generate_financial_report = Tool(
    name="generate_financial_report",
    description="""
        Generates a comprehensive financial report as of a specified date, including:
        - Current cash balance
        - Inventory valuation
        - Combined asset total
        - Itemized inventory breakdown
        - Top 5 best-selling products
    """,
    function=generate_financial_report,
    require_parameter_descriptions=True
)

tool_search_quote_history = Tool(
    name="search_quote_history",
    description="""
        Searches historical quotes for keywords in customer requests or quote explanations, returning results sorted by most recent order date and limited by the specified `limit` parameter.
    """,
    function=search_quote_history,
    require_parameter_descriptions=True
)


# Set up your agents and create an orchestration agent that will manage them.
inventory_agent_toolset = [tool_get_all_inventory, tool_get_stock_level, tool_get_supplier_delivery_date, tool_create_transaction]
quoting_agent_toolset = [tool_search_quote_history, tool_generate_financial_report]
ordering_agent_toolset = [tool_create_transaction, tool_get_supplier_delivery_date]
business_advisory_agent_toolset = [tool_generate_financial_report, tool_get_cash_balance, tool_search_quote_history]


# Define output model for the orchestration agent
class OrchestrationClassification(BaseModel):
    classification: Literal["INQUIRY", "ORDER"]


class InventoryResponse(BaseModel):
    answer: str
    proceed: bool


# Definee the orchestration agent
orchestration_agent = Agent(model=model,
                            name="Orchestration Agent",
                            model_settings=ModelSettings(temperature=0.0),
                            system_prompt="""
                                You are the Orchestration Agent for Beaver's Choice Paper Company, responsible for classifying incoming customer requests into one of two categories: **INQUIRY** or **ORDER**.

                                ### Classification Rules:
                                
                                1. **INQUIRY**: Classify as INQUIRY if the customer is seeking information, such as:
                                - Availability, stock levels, or delivery dates.
                                - Price comparisons or historical quotes without intent to purchase.
                                2. **ORDER**: Classify as ORDER if the customer expresses intent to:
                                - Purchase, order, or buy a product (even without specifying quantities).
                                - Finalize or proceed with a purchase.
                                3. **Default**: If unsure, classify as **INQUIRY**.

                                ### Output Format:
                                
                                Return a JSON object adhering to the following Pydantic schema:
                                ```python
                                class OrchestrationClassification(BaseModel):
                                    classification: Literal["INQUIRY", "ORDER"]
                                ```
                                
                                Examples:

                                INQUIRY: "Do you have A4 paper in stock?"
                                ORDER: "I’d like to buy 100 units of copier paper."

                                Focus solely on classification and ensure the output strictly follows the specified JSON format.
                                """,
                            output_type=OrchestrationClassification
                            )

# Defines the inventory agent
inventory_agent = Agent(model=model,
                       name="Inventory Agent",
                       model_settings=ModelSettings(temperature=0.3),
                       system_prompt="""
                            You are the Inventory Agent for Beaver's Choice Paper Company, responsible for handling structured requests classified as either **INQUIRY** or **ORDER**. Follow the appropriate logic based on the classification and use the available tools to generate accurate responses.

                            ### Decision Logic:
                            
                            #### **IF classification == "INQUIRY"**:
                            1. Check stock levels using `get_stock_level` for the requested item(s).  
                            2. If the customer asks about delivery feasibility, use `get_supplier_delivery_date` to estimate availability.  
                            3. Provide a clear, helpful response with available quantity and, if relevant, the estimated delivery date.  
                            4. **DO NOT** trigger any inventory changes.  

                            #### **IF classification == "ORDER"**:
                            1. Check stock levels using `get_stock_level` for the requested item(s).  
                            2. **If stock is sufficient**: Confirm the order can be fulfilled immediately.  
                            3. **If stock is insufficient**:  
                            - Use `create_transaction` to initiate a restocking order.  
                            - Use `get_supplier_delivery_date` to estimate restocking time.  
                            - Inform the next agent that the material has been reordered.  
                            - If the supplier delivery date is **after** the expected delivery date, set `proceed` to `False` and explain to the customer that the order cannot be fulfilled immediately.  

                            ### Output Expectations:
                            - Clearly state whether stock is sufficient or not.  
                            - If a restocking order was triggered, include the expected delivery date and the outcome of `get_supplier_delivery_date`.  
                            - Be accurate, concise, and customer-friendly.  

                            ### Tools Available:
                            - `get_all_inventory`: Retrieve a full inventory snapshot.  
                            - `get_stock_level`: Check quantity for a specific item.  
                            - `get_supplier_delivery_date`: Estimate restocking delivery time.  
                            - `create_transaction`: Place a restocking order (only for **ORDER** requests).  

                            ### Output Format:
                            Return a JSON object adhering to the following Pydantic schema:
                            ```python
                            class InventoryResponse(BaseModel):
                                answer: str
                                proceed: bool
                            ```
                            
                            Examples:

                            INQUIRY: Customer asks about A4 paper stock.
                            Response: {"answer": "We have 500 units of A4 paper in stock.", "proceed": false}

                            ORDER: Customer orders 1000 units of A4 paper (only 500 in stock).
                            Response: {"answer": "The item has been reordered. Expected delivery: 2023-11-15.", "proceed": false}

                            Always follow this logic and ensure the output strictly adheres to the specified JSON format.
                            """,
                       tools=inventory_agent_toolset,
                       output_type=InventoryResponse
                       )

# Defines the quoting agent
quoting_agent = Agent(model=model,
                      name="Quoting Agent",
                      model_settings=ModelSettings(temperature=0.7),
                      system_prompt="""
                        You are the Quoting Agent for Beaver's Choice Paper Company, responsible for generating competitive and strategic sales quotes based on:  
                        - The customer's order request.  
                        - Inventory and delivery information provided by the Inventory Agent.  
                        - Historical quote and sales data.  

                        ### Step-by-Step Responsibilities:
                        1. **Analyze the Customer's Request**:  
                        - Identify the requested item(s), quantity, and delivery expectations.  

                        2. **Use Inventory Context**:  
                        - Rely on provided inventory and delivery information (e.g., availability, delivery dates).  
                        - **Do not** check inventory independently - this has already been handled.  

                        3. **Analyze Pricing History**:  
                        - Use `search_quote_history` to find comparable past quotes.  
                        - Use `generate_financial_report` to identify pricing trends or profitability patterns if needed.  

                        4. **Calculate a Competitive Quote**:  
                        - Apply volume discounts for large orders.  
                        - Factor in urgency, customer history (if available), and market alignment.  
                        - Balance profitability with customer attractiveness.  

                        5. **Prepare the Output**:  
                        - Provide a clear price per unit and total price.  
                        - Include remarks if applicable (e.g., "Discount applied for high volume").  

                        ### Tools Available:
                        - `search_quote_history`: Retrieve past quotes for reference.  
                        - `generate_financial_report`: Analyze sales trends and pricing patterns.  

                        ### Output Expectations:
                        - **Price per unit** and **total price** must be clearly stated.  
                        - Include remarks to justify discounts or special considerations.  
                        - Ensure the quote is competitive, profitable, and aligned with business goals.  

                        ### Examples:
                        
                        - **Request**: 500 units of A4 paper.  
                        **Response**:  

                        Price per unit : $0.10
                        Total price    : $50.00
                        Remarks        : Standard pricing applied.

                        - **Request**: 5000 units of A4 paper.  
                        **Response**:  

                        Price per unit : $0.09
                        Total price    : $450.00
                        Remarks        : Volume discount applied.

                        Focus solely on generating optimized quotes based on the provided information and business objectives.
                        """,
                      tools=quoting_agent_toolset
                      )

# Define ordering agent
ordering_agent  = Agent(model,
                        name="Ordering Agent",
                        model_settings=ModelSettings(temperature=0.5),
                        system_prompt="""
                            You are the Ordering Agent for Beaver's Choice Paper Company, responsible for completing customer orders based on the provided quote and current inventory status.  

                            ### Responsibilities:
                            1. **Assume Customer Acceptance**:  
                            Proceed with the order as if the customer has accepted the quote. No confirmation is needed.  

                            2. **Estimate Delivery Date**:  
                            Calculate the delivery date based on:  
                            - Order size.  
                            - Current date.  
                            - Use `get_supplier_delivery_date` if necessary.  

                            3. **Record the Sale**:  
                            Use `create_transaction` to store the order details, including:  
                            - Item name(s).  
                            - Quantity.  
                            - Price per unit.  
                            - Total price.  
                            - Order date.  

                            4. **Respond to the Customer**:  
                            - Confirm the order was successful.  
                            - Provide the estimated delivery date.  
                            - Thank the customer for their business.  

                            ### Tools Available:
                            - `get_supplier_delivery_date`: Estimate delivery time for out-of-stock items.  
                            - `create_transaction`: Finalize and save the sale in the system.  

                            ### Important Notes:
                            - **Do not** generate a new quote or modify pricing—this has already been handled.  
                            - Focus solely on verifying feasibility and executing the transaction.  
                            - Maintain a polite, professional, and customer-friendly tone.  

                            ### Output Expectations:
                            Provide a clear and concise response to the customer, including:  
                            1. Order confirmation.  
                            2. Estimated delivery date.  
                            3. A thank-you message.  

                            ### Example Response:

                            "Your order has been successfully placed. Estimated delivery date: 2023-11-20. Thank you for choosing Beaver's Choice Paper Company!"

                            Always ensure the transaction is recorded accurately and the customer is informed professionally.
                            """,
                        tools=ordering_agent_toolset
                        )

# Define invoicing agent
# Note: toolless agent
invoice_agent = Agent(model=model,
                      name="Invoicing Agent",
                      model_settings=ModelSettings(temperature=0.3),
                      system_prompt="""
                        You are the Invoicing Agent for Beaver's Choice Paper Company, responsible for generating professional and complete customer invoices based on finalized orders.  

                        ### Input Data:
                        You receive structured data including:  
                        - Customer name and optional contact information (address, email).  
                        - Item(s), quantities, unit prices, and total price.  
                        - Discounts applied (if any).  
                        - Delivery date (if known).  

                        ### Response Requirements:
                        Your response consists of **two parts**:  
                        1. **Friendly Response Text**:  
                        - Thank the customer for their order.  
                        - Confirm the ordered items and delivery date.  
                        - Mention that the invoice is attached below.  

                        2. **Formatted Invoice (Plain Text)**:  
                        Generate a well-formatted `.txt` invoice block using ASCII layout. Include the following details:  
                        - **Invoice Number**: Realistic placeholder (e.g., `INV-2025-XXX`).  
                        - **Date of Issue**: Current date.  
                        - **Customer Details**: Name, address, and email (use `<placeholder>` if missing).  
                        - **Itemized List**: Name, quantity, unit price, and line total.  
                        - **Total Amount**: Net total, discount (if applicable), and grand total.  
                        - **Delivery Date**.  
                        - **Thank You Note** at the bottom.  

                        ### Formatting Notes:
                        - Use a **monospaced layout** for the invoice block.  
                        - Align columns with **spaces** (not tabs).  
                        - Keep the width readable (max **80 characters**).  
                        - Separate sections with dashed lines (`-----`) or whitespace.  

                        ### Example Invoice (Shortened):

                        ```txt
                        Invoice No: INV-2026-001
                        Date: 2026-02-15

                        Bill/Ship To:
                        Name:    John Doe
                        Address: <placeholder>
                        Email:   john.doe@mail.com

                        Items:
                        Qty   Description          Unit Price    Line Total
                        1000  A4 Paper (80g/m²)         $0.10       $100.00

                        Subtotal:                                   $100.00
                        Discount (10%):                             -$10.00
                        Total Amount Due:                            $90.00

                        Expected Delivery Date: 2026-02-27

                        Thank you for shopping with us!
                        ```
                        
                        ### Mandatory Requirements:
                        - Always generate a **full invoice** with all required details.  
                        - Explicitly list any discounts applied.  
                        - Use `<placeholder>` for missing customer information.  
                        - Maintain a **clear, professional tone**.  

                        ### Example Friendly Response Text:
                        
                        "Thank you for your order, John Doe! We confirm your purchase of 1000 units of A4 Paper (80g/m²), scheduled for delivery on 2026-02-27. Please find your invoice below."  

                        Always ensure the invoice is accurate, well-formatted, and professional.
                        """)

# Define the business advisory agent
business_advisory_agent = Agent(model=model,
                                name="Business Advisory Agent",
                                model_settings=ModelSettings(temperature=0.5),
                                system_prompt="""
                                    You are the Business Advisory Agent for Beaver's Choice Paper Company, responsible for analyzing transactions and providing actionable recommendations to improve business efficiency and revenue.

                                    ### Responsibilities:
                                    1. **Analyze Transactions**: Review the financial report and transaction history to identify trends, inefficiencies, or opportunities.
                                    2. **Identify Opportunities**: Look for patterns such as high-demand products, low-margin items, or inventory overstock/understock.
                                    3. **Provide Recommendations**: Suggest changes to business operations, such as adjusting inventory levels, promoting certain products, or optimizing pricing strategies.
                                    4. **Focus on Efficiency and Revenue**: Ensure recommendations are aimed at increasing profitability and operational efficiency.

                                    ### Tools Available:
                                    - `generate_financial_report`: Analyze sales trends and pricing patterns.   
                                    - `get_cash_balance`: FCalculate the current cash balance as of a specified date.
                                    - `search_quote_history`: Retrieve past quotes for reference.

                                    ### Output Expectations:
                                    - Provide clear, concise, and actionable recommendations.
                                    - Highlight key areas for improvement and potential impact on revenue or efficiency.
                                    - Use a professional and data-driven tone.

                                    ### Example Output:
                                    "Recommendation: Increase inventory of A4 paper by 20% due to high demand and consistent sales. Consider promoting glossy paper to reduce excess stock. Adjust pricing for cardstock to improve profit margins."

                                    Always ensure recommendations are based on data and aligned with business goals.
                                """,
                                tools=business_advisory_agent_toolset
                            )
class WorkflowContext(BaseModel):
    """Shared context between agents"""
    request_id: str
    request_body: str
class MultiAgentWorkflow:
    def __init__(self):
        self.agents = {
            "orchestration": orchestration_agent,
            "inventory": inventory_agent,
            "quoting": quoting_agent,
            "sales": ordering_agent,
            "invoice": invoice_agent,
            "business_advisory": business_advisory_agent
        }

        self.agent_usage_count = {
            "inventory": 0,
            "quoting": 0,
            "sales": 0,
            "invoice": 0,
            "business_advisory": 0
        }

    def handle_inquiry(self, context: WorkflowContext) -> str:
        """
        Handle customer inquiries by using the quoting agent to generate a financial report and the evaluation agent to assess the response.

        Parameters:
        - context (WorkflowContext): The context of the inquiry including original request.
        
        Returns:
        - str: The output from the evaluation agent after assessing the generated financial report.
        """
        # Call quoting agent to generate financial report
        prompt = f"""
        Classification: INQUIRY

        User Request: {context.request_body}
        """
        try:
            inventory_response = self.agents["inventory"].run_sync(
                prompt,
                deps=context
            )
            self.agent_usage_count["inventory"] += 1
            return inventory_response.output
        except UsageLimitExceeded as e:
            logger.error(f"Usage Limit exceeded calling inventory agent: {e}")
            return "Usage Limit exceeded calling inventory agent."

    def handle_order(self, context: WorkflowContext) -> str:
        """
        Handle an order by orchestrating interactions with various agents to process the request.

        Parameters:
            context (WorkflowContext): The workflow context containing original user request and dependencies.

        Returns:
            str: Final response from the sales agent after processing all steps.
        """

        inventory_prompt = f"""
            Classification: ORDER
        
            User Request: {context.request_body}
        """
        
        # Call inventory agent to check stock levels and handle order for stock items
        try:
            inventory_response = self.agents["inventory"].run_sync(
                inventory_prompt,
                deps=context
            )
            self.agent_usage_count["inventory"] += 1
        except UsageLimitExceeded as e:
            logger.error(f"Usage Limit exceeded calling inventory agent: {e}")
            return "Usage Limit exceeded calling inventory agent"

        if not inventory_response.output.proceed:
            # If inventory agent indicates order cannot proceed, return a message
            logger.warning(f"Order cannot be processed: {inventory_response.output.answer}")
            print(f"Order cannot be processed: {inventory_response.output.answer}")
            return inventory_response.output.answer

        quote_prompt = f"""
            User Request: {context.request_body}
            Inventory Context: {inventory_response.output.answer}
        """
        
        # Call quoting agent to generate a quote based on the order
        try:
            quoting_response = self.agents["quoting"].run_sync(
                quote_prompt,
                deps=context
            )
        except UsageLimitExceeded as e:
            logger.error(f"Usage Limit exceeded calling quoting agent: {e}")
            return "Usage Limit exceeded calling quote agent"

        self.agent_usage_count["quoting"] += 1

        sales_prompt = f"""
            User Request: {context.request_body}
            Inventory Context: {inventory_response.output.answer}
            Quote Context: {quoting_response.output}
        """
        
        # Call sales finalization agent to finalize the order
        try:
            sales_response = self.agents["sales"].run_sync(
                sales_prompt,
                deps=context
            )
        except UsageLimitExceeded as e:
            logger.error(f"Usage Limit exceeded calling sales agent: {e}")
            return "Usage Limit exceeded calling sales agent"
        
        self.agent_usage_count["sales"] += 1

        invoice_prompt = f"""
            User Request: {context.request_body}
            Inventory Context: {inventory_response.output.answer}
            Quote Context: {quoting_response.output}
            Sales Context: {sales_response.output}
        """
        
        # Call invoice agent to generate an invoice for the order
        try:
            invoice_response = self.agents["invoice"].run_sync(
                invoice_prompt,
                deps=context
            )
            self.agent_usage_count["invoice"] += 1
            return invoice_response.output
        except UsageLimitExceeded as e:
            logger.error(f"Usage Limit exceeded calling invoice agent: {e}")
            return "Usage Limit exceeded calling invoice agent"

    def handle_advisory_report(self, context: WorkflowContext) -> str:
        """
        Generates and prints business advisory recommendations based on the current financial report.

        Args:
            context (WorkflowContext): The workflow context containing request information.
        """
        try:
            advisory_response = self.agents["business_advisory"].run_sync(
                f"Financial report: {generate_financial_report(datetime.now().isoformat())}",
                deps=context
            )
            self.agent_usage_count["business_advisory"] += 1
            return advisory_response.output
        except UsageLimitExceeded as e:
            logger.error(f"Usage Limit exceeded calling business advisory agent: {e}")
            print("Usage Limit exceeded calling business advisory agent.")
            return ""

    def run(self, customer_request: str) -> str:
        """
        Orchestrates multi-agent workflow for a given customer request.
        
        This method creates a workflow context and orchestrates agents to handle the request,
        ultimately returning a response based on the classification provided by the orchestration agent.

        Parameters:
            customer_request (str): The customer's request that needs to be handled.

        Returns:
            str: A response from the multi-agent workflow, indicating the outcome of handling the request.
        
        Raises:
            Exception: If an unexpected error occurs during the execution of the workflow.
        """

        # Create workflow context
        self.workflow_context = WorkflowContext(
            request_id=f"REQ_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            request_body=customer_request
        )

        # Step 1: Call the orchestration agent to classify the request
        try:
            orchestration_response = self.agents["orchestration"].run_sync(
                customer_request,
                deps=self.workflow_context
            )
        except UsageLimitExceeded as e:
            logger.error(f"Usage Limit exceeded calling orchestration agent: {e}")
            return "Usage Limit exceeded calling orchestration agent"
        
        print(f"Orchestration Agent classified request as: {orchestration_response.output.classification}")

        # Step 2: Based on classification, route to appropriate agents
        if orchestration_response.output.classification == "INQUIRY":
            # Handle inquiries with quoting and evaluation agents
            response = self.handle_inquiry(self.workflow_context)
        elif orchestration_response.output.classification == "ORDER":
            # Handle orders with inventory and sales agents
            response = self.handle_order(self.workflow_context)
        else:
            response = "Invalid classification received from orchestration agent."

        return response

# Run your test scenarios by writing them here. Make sure to keep track of them.
def run_test_scenarios():
    """
    Initializes a database and processes sample quote requests data.

    This function performs the following steps:
    1. Logs initialization of the Database.
    2. Calls `init_database` with the specified db_engine.
    3. Reads the CSV file "quote_requests_sample.csv".
    4. Converts the 'request_date' column to datetime format, handling errors by coercing them into NaT (Not a Time).
    5. Drops rows where 'request_date' is missing.
    6. Sorts the DataFrame by 'request_date'.
    7. Reads "quote_requests_sample.csv" again for consistency.
    8. Converts the 'request_date' column to datetime format and sorts the DataFrame by this column.
    9. Determines the initial date from the earliest request_date in the sample data.
    10. Generates a financial report using the initial date and stores it in `report`.
    11. Extracts the current cash balance and inventory value from the generated report.

    If any error occurs during these steps, logs an error message and returns early.
    """
    logger.info("Initializing Database...")
    init_database(db_engine)
    
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        logger.error(f"FATAL: Error loading test data: {e}")
        return

    # Read the CSV file again for consistency
    quote_requests_sample = pd.read_csv("quote_requests_sample.csv")

    # Convert 'request_date' to datetime and sort by this column
    quote_requests_sample["request_date"] = pd.to_datetime(quote_requests_sample["request_date"])
    quote_requests_sample = quote_requests_sample.sort_values("request_date")

    # Determine the initial date from the earliest request_date in the sample data
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    
    # Generate a financial report using the initial date and store it in `report`
    financial_report = generate_financial_report(initial_date)
    
    # Extract current cash balance and inventory value from the generated report
    cash_balance = financial_report["cash_balance"]
    inventory_value = financial_report["inventory_value"]

    ###########################################
    # INITIALIZE YOUR MULTI AGENT SYSTEM HERE #
    ###########################################

    multi_agent_workflow = MultiAgentWorkflow()

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(' ' * 80)
        print(f"===== Request {idx + 1} =====")
        print(' ' * 80)
        print(f"Context         : {row['job']} organizing {row['event']}")
        print(f"Request Date    : {request_date}")
        print(f"Cash Balance    : ${cash_balance:.2f}")
        print(f"Inventory Value : ${inventory_value:.2f}")

        # Process request
        full_request = f"{row['request']} (Date of request: {request_date})"

        #####################################################
        # USE YOUR MULTI AGENT SYSTEM TO HANDLE THE REQUEST #
        #####################################################

        response = multi_agent_workflow.run(full_request)

        # Update state
        financial_report = generate_financial_report(request_date)
        cash_balance = financial_report["cash_balance"]
        inventory_value = financial_report["inventory_value"]

        print(f"Response:          {response}")
        print(f"Updated Cash:      ${cash_balance:.2f}")
        print(f"Updated Inventory: ${inventory_value:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": cash_balance,
                "inventory_value": inventory_value,
                "response": response,
            }
        )

        # Sleeps for 100ms
        time.sleep(0.1)

    #############################################################################################
    # Final report
    # Includes final financial report, business advisory recommendations and agent call summary
    #############################################################################################
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)

    print('=' * 80)
    print("============================ FINAL FINANCIAL REPORT ============================")
    print('=' * 80)
    print(f"Final Cash:      ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Call business advisory agent for final recommendations
    final_recommendations = multi_agent_workflow.handle_advisory_report(multi_agent_workflow.workflow_context)
    print('=' * 80)
    print("====================== BUSINESS ADVISORY RECOMMENDATIONS =======================")
    print('=' * 80)
    print(final_recommendations)

    # Print the agents' calls summary
    print('=' * 80)
    print("============================== AGENT CALL SUMMARY ==============================")
    print('=' * 80)
    for agent, count in multi_agent_workflow.agent_usage_count.items():
        print(f"{agent.capitalize()} Agent: {count} times")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results


if __name__ == "__main__":
    results = run_test_scenarios()
