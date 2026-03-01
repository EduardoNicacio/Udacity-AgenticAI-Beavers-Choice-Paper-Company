"""
Beaver's Choice Paper Company - Multi-Agent Workflow System

This module implements a simple inventory, quoting, ordering, and invoicing workflow for a paper company.
It uses SQLAlchemy with an SQLite database, pandas for data manipulation, and the pydantic_ai library to orchestrate multiple LLM-powered agents.

The core components are:
- A set of utility functions that interact with the database (inventory, transactions, cash balance, etc.).
- Five agents: Orchestration Agent, Inventory Agent, Quoting Agent, Ordering Agent, Invoicing Agent.
- The MultiAgentWorkflow class that coordinates the agents based on a customer's request.

The script also contains a `run_test_scenarios` function which demonstrates how to initialize the database and process a batch of sample quote requests.
"""

import ast
import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

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
    {
        "item_name": "Paper plates",
        "category": "product",
        "unit_price": 0.10,
    },  # per plate
    {"item_name": "Paper cups", "category": "product", "unit_price": 0.08},  # per cup
    {
        "item_name": "Paper napkins",
        "category": "product",
        "unit_price": 0.02,
    },  # per napkin
    {
        "item_name": "Disposable cups",
        "category": "product",
        "unit_price": 0.10,
    },  # per cup
    {
        "item_name": "Table covers",
        "category": "product",
        "unit_price": 1.50,
    },  # per cover
    {
        "item_name": "Envelopes",
        "category": "product",
        "unit_price": 0.05,
    },  # per envelope
    {
        "item_name": "Sticky notes",
        "category": "product",
        "unit_price": 0.03,
    },  # per sheet
    {"item_name": "Notepads", "category": "product", "unit_price": 2.00},  # per pad
    {
        "item_name": "Invitation cards",
        "category": "product",
        "unit_price": 0.50,
    },  # per card
    {"item_name": "Flyers", "category": "product", "unit_price": 0.15},  # per flyer
    {
        "item_name": "Party streamers",
        "category": "product",
        "unit_price": 0.05,
    },  # per roll
    {
        "item_name": "Decorative adhesive tape (washi tape)",
        "category": "product",
        "unit_price": 0.20,
    },  # per roll
    {
        "item_name": "Paper party bags",
        "category": "product",
        "unit_price": 0.25,
    },  # per bag
    {
        "item_name": "Name tags with lanyards",
        "category": "product",
        "unit_price": 0.75,
    },  # per tag
    {
        "item_name": "Presentation folders",
        "category": "product",
        "unit_price": 0.50,
    },  # per folder
    # Large-format items (priced per unit)
    {
        "item_name": "Large poster paper (24x36 inches)",
        "category": "large_format",
        "unit_price": 1.00,
    },
    {
        "item_name": "Rolls of banner paper (36-inch width)",
        "category": "large_format",
        "unit_price": 2.50,
    },
    # Specialty papers
    {"item_name": "100 lb cover stock", "category": "specialty", "unit_price": 0.50},
    {"item_name": "80 lb text paper", "category": "specialty", "unit_price": 0.40},
    {"item_name": "250 gsm cardstock", "category": "specialty", "unit_price": 0.30},
    {"item_name": "220 gsm poster paper", "category": "specialty", "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system


def generate_sample_inventory(
    paper_supplies: list, coverage: float = 0.4, seed: int = 11235
) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` x N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
            keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 11235).

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
        range(len(paper_supplies)), size=num_items, replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append(
            {
                "item_name": item["item_name"],
                "category": item["category"],
                "unit_price": item["unit_price"],
                "current_stock": np.random.randint(1000, 5000),
                "min_stock_level": np.random.randint(250, 500),
            }
        )

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)


def init_database(db_engine: Engine, seed: int = 11235) -> Engine:
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
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels. Default is 11235.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame(
            {
                "id": [],
                "item_name": [],
                "transaction_type": [],  # 'stock_orders' or 'sales'
                "units": [],  # Quantity involved
                "price": [],  # Total price for the transaction
                "transaction_date": [],  # ISO-formatted date
            }
        )
        transactions_schema.to_sql(
            "transactions", db_engine, if_exists="replace", index=False
        )

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql(
            "quote_requests", db_engine, if_exists="replace", index=False
        )

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
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(
                lambda x: x.get("job_type", "")
            )
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(
                lambda x: x.get("order_size", "")
            )
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(
                lambda x: x.get("event_type", "")
            )

        # Retain only relevant columns
        quotes_df = quotes_df[
            [
                "request_id",
                "total_amount",
                "quote_explanation",
                "order_date",
                "job_type",
                "order_size",
                "event_type",
            ]
        ]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append(
            {
                "item_name": None,
                "transaction_type": "sales",
                "units": None,
                "price": 50000.0,
                "transaction_date": initial_date,
            }
        )

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append(
                {
                    "item_name": item["item_name"],
                    "transaction_type": "stock_orders",
                    "units": item["current_stock"],
                    "price": item["current_stock"] * item["unit_price"],
                    "transaction_date": initial_date,
                }
            )

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql(
            "transactions", db_engine, if_exists="append", index=False
        )

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
    This function records a transaction of type 'stock_orders' or 'sales' with specified
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
        transaction = pd.DataFrame(
            [
                {
                    "item_name": item_name,
                    "transaction_type": transaction_type,
                    "units": quantity,
                    "price": round(price, 2),
                    "transaction_date": date_str,
                }
            ]
        )

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
    print(
        f"FUNC (get_stock_level): Fetching stock for '{item_name}' as of '{as_of_date}'"
    )

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
        - 11-100 units: 1 day
        - 101-1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(
        f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'"
    )

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(
            f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base."
        )
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
            total_sales = transactions.loc[
                transactions["transaction_type"] == "sales", "price"
            ].sum()
            total_purchases = transactions.loc[
                transactions["transaction_type"] == "stock_orders", "price"
            ].sum()
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
    cash_balance = get_cash_balance(as_of_date)

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

        inventory_summary.append(
            {
                "item_name": item["item_name"],
                "stock": stock,
                "unit_price": item["unit_price"],
                "value": item_value,
            }
        )

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
        "cash_balance": cash_balance,
        "inventory_value": inventory_value,
        "total_assets": cash_balance + inventory_value,
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
from pydantic_ai import Agent, UsageLimitExceeded, UsageLimits
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.tools import Tool
from pydantic_ai.settings import ModelSettings


# Set up and load your env parameters and instantiate your model.
load_dotenv()

# Create a model instance with a custom API URL
# This could be a self-hosted OpenAI-compatible API, Azure endpoint, or proxy
# Note: OpenAIChatModel -> AsyncOpenAI -> reads OPENAI_API_KEY and OPENAI_BASE_URL from environment variables
#       OPENAI_API_MODEL is being retrieved from the environment variables, with fallback to ""
model = OpenAIChatModel(model_name=os.getenv("OPENAI_API_MODEL", "gpt-4o-mini"))


# Define tools for the agents
tool_create_transaction = Tool(
    name="create_transaction",
    description="""
        Records a new transaction (`stock_orders` or `sales`) with specified item name, transaction_type, quantity, total price, and transaction date into the database's transactions table.
    """,
    function=create_transaction,
    require_parameter_descriptions=True,
)

tool_get_all_inventory = Tool(
    name="get_all_inventory",
    description="""
        Retrieves a snapshot of available inventory as of a specified date by calculating net stock (sum of stock orders minus sales) for all items with positive quantities.
    """,
    function=get_all_inventory,
    require_parameter_descriptions=True,
)

tool_get_stock_level = Tool(
    name="get_stock_level",
    description="""
        Retrieves the current stock level for a specified item as of a given date.
    """,
    function=get_stock_level,
    require_parameter_descriptions=True,
)

tool_get_supplier_delivery_date = Tool(
    name="get_supplier_delivery_date",
    description="""
        Estimates delivery date based on order quantity and starting date, with lead times determined by order size:
        - Up to 10 units: same day
        - 11-100 units: 1 day
        - 101-1000 units: 4 days
        - Over 1000 units: 7 days
    """,
    function=get_supplier_delivery_date,
    require_parameter_descriptions=True,
)

tool_get_cash_balance = Tool(
    name="get_cash_balance",
    description="""
        Calculates cash balance as of a specified date by subtracting total stock purchase costs (from stock_orders) from total revenue (from sales) in the transactions table up to that date.
    """,
    function=get_cash_balance,
    require_parameter_descriptions=True,
)

tool_generate_financial_report = Tool(
    name="generate_financial_report",
    description="""
        Generates a comprehensive financial report as of a specific date, including:
        - Cash balance
        - Inventory valuation
        - Combined asset total
        - Itemized inventory breakdown
        - Top 5 best-selling products
    """,
    function=generate_financial_report,
    require_parameter_descriptions=True,
)

tool_search_quote_history = Tool(
    name="search_quote_history",
    description="""
        Searches historical quotes for keywords in customer requests or quote explanations, returning results sorted by most recent order date and limited by the specified `limit` parameter.
    """,
    function=search_quote_history,
    require_parameter_descriptions=True,
)


# Set up your agents and create an orchestration agent that will manage them.
inventory_agent_toolset = [
    tool_get_all_inventory,
    tool_get_cash_balance,
    tool_get_stock_level,
    tool_get_supplier_delivery_date,
    tool_create_transaction,
]
quoting_agent_toolset = [tool_search_quote_history, tool_generate_financial_report]
ordering_agent_toolset = [
    tool_get_cash_balance,
    tool_create_transaction,
    tool_get_supplier_delivery_date,
]


# Defines the Orchestration Agent
class OrchestrationClassification(BaseModel):
    """
    Pydantic model used by the orchestration agent to classify a customer request.

    Attributes:
        classification (Literal["INQUIRY", "ORDER"]): The determined type of request.
    """

    classification: Literal["INQUIRY", "ORDER"]


class InventoryResponse(BaseModel):
    """
    Response format from the Inventory Agent.

    Attributes:
        answer (str): Human-readable message to return to the user.
        proceed (bool): Indicates whether the order can be processed immediately.
        rationale (Optional[str]): Optional explanation when `proceed` is False.
    """

    answer: str
    proceed: bool
    rationale: Optional[str] = None


# Defines the Orchestration Agent
orchestration_agent = Agent(
    model=model,
    name="Orchestration Agent",
    model_settings=ModelSettings(temperature=0.0),
    system_prompt="""
        You are the Orchestration Agent for Beaver's Choice Paper Company, responsible for classifying incoming customer requests into one of two categories: **INQUIRY** or **ORDER**.

        ### Classification Rules:
        1. **INQUIRY**: Customer seeks information (availability, stock levels, delivery dates, price comparisons, historical quotes) without intent to purchase.
        2. **ORDER**: Customer expresses intent to purchase, order, or buy a product.
        3. **Default**: If unsure, classify as **INQUIRY**.

        ### Output Format:
        Return a JSON object adhering to the following Pydantic schema:
        ```python
        class OrchestrationClassification(BaseModel):
            classification: Literal["INQUIRY", "ORDER"]
        ```
    """,
    output_type=OrchestrationClassification,
)

# Defines the Inventory Agent
inventory_agent = Agent(
    model=model,
    name="Inventory Agent",
    model_settings=ModelSettings(temperature=0.1),
    system_prompt="""
        You are the Inventory Agent for Beaver's Choice Paper Company, responsible for handling structured requests classified as either **INQUIRY** or **ORDER**. Follow the appropriate logic based on the classification and use the available tools to generate accurate responses.

        ### Decision Logic:
        #### Convert all dates not in ISO8601 format to this format before calling any tool.
        Examples: 'April 15, 2025' → '2025-04-15', 'July 4, 2025' → '2025-07-04'.

        #### IF classification == "INQUIRY":
        1. Check stock levels using `get_stock_level` for the requested item(s).  
        2. If delivery feasibility is asked, use `get_supplier_delivery_date`.  
        3. Provide a clear response with quantity and estimated delivery date if relevant.  
        4. **DO NOT** trigger any inventory changes.

        #### IF classification == "ORDER":
        1. Check stock levels using `get_stock_level` for the requested item(s).  
        2. If stock is sufficient: confirm fulfillment immediately.  
        3. If stock insufficient:
            a. Retrieve current cash balance via `get_cash_balance`.  
                - If cash < estimated restock cost, set `proceed` to False and explain.  
            b. If cash >= estimated restock cost:
                i. Use `create_transaction` to reorder the item.  
                ii. Estimate restocking time with `get_supplier_delivery_date`.   
        4. Inform next agent that material has been reordered if applicable.

        ### Output Expectations:
        - State whether stock is sufficient.
        - Include expected delivery date if a restocking order was triggered.
        - Be concise, customer-friendly.

        ### Tools Available:
        - `get_all_inventory`
        - `get_cash_balance`
        - `get_stock_level`
        - `get_supplier_delivery_date`
        - `create_transaction`

        ### Output Format:
        Return a JSON object adhering to the following Pydantic schema:
        ```python
        class InventoryResponse(BaseModel):
            answer: str
            proceed: bool
            rationale: Optional[str] = None
        ```
    """,
    tools=inventory_agent_toolset,
    output_type=InventoryResponse,
)

# Defines the Quoting Agent
quoting_agent = Agent(
    model=model,
    name="Quoting Agent",
    model_settings=ModelSettings(temperature=0.3),
    system_prompt="""
        You are the Quoting Agent for Beaver's Choice Paper Company, responsible for generating competitive and strategic sales quotes based on:
        - The customer's order request.
        - Inventory and delivery information provided by the Inventory Agent.
        - Historical quote and sales data.

        ### Responsibilities:
        1. Identify requested items, quantity, and delivery expectations from the customer's request.
        2. Use inventory context (availability, delivery dates) - do not check inventory again.
        3. Retrieve comparable past quotes with `search_quote_history`.
        4. Apply volume discounts for large orders; factor in urgency, customer history, and market alignment.
        5. Provide clear price per unit, total price, and remarks justifying any discount or special consideration.

        ### Tools Available:
        - `search_quote_history`
        - `generate_financial_report`

        ### Output Expectations:
        - Clearly state price per unit and total price.
        - Include remarks to justify discounts or special considerations.
    """,
    tools=quoting_agent_toolset,
)

# Defines the Ordering Agent
ordering_agent = Agent(
    model=model,
    name="Ordering Agent",
    model_settings=ModelSettings(temperature=0.3),
    system_prompt="""
        You are the Ordering Agent for Beaver's Choice Paper Company, responsible for completing customer orders based on the provided quote and current inventory status.

        ### Responsibilities:
        1. Assume customer acceptance - proceed with the `sales` order.
        2. Estimate delivery date using order size, current date, and `get_supplier_delivery_date` if needed.
        3. Record the order with `create_transaction`.
        4. Respond to the customer: confirm success, provide estimated delivery date, thank them.

        ### Tools Available:
        - `get_cash_balance`
        - `get_supplier_delivery_date`
        - `create_transaction`

        ### Important Notes:
        - Do not generate a new quote or modify pricing.
        - Do not create a `stock_orders` order if cash balance is negative.
        - If restocking isn't possible, inform the customer.

        ### Output Expectations:
        Provide a clear and concise response to the customer, including:
        1. Order confirmation.
        2. Estimated delivery date.
        3. Thank-you message.
    """,
    tools=ordering_agent_toolset,
)

# Define the Invoicing Agent
invoice_agent = Agent(
    model=model,
    name="Invoicing Agent",
    model_settings=ModelSettings(temperature=0.1),
    system_prompt="""
                    You are the Invoicing Agent for Beaver's Choice Paper Company, responsible for generating professional and complete customer invoices based on finalized orders.

                    ### Input Data:
                    Structured data including customer details, items, quantities, unit prices, total price, discounts (if any), and delivery date.

                    ### Response Requirements:
                    1. Friendly response text: thank the customer, confirm items and delivery date, mention invoice attachment.
                    2. Formatted plain-text invoice block using ASCII layout with columns aligned by spaces (max width 80).

                    ### Formatting Notes:
                    - Monospaced layout; align columns with spaces.
                    - Separate sections with dashed lines or whitespace.
                    
                    ### Example Invoice:

                    ```txt
                    Invoice No: INV-2026-001
                    Date: 2026-02-15

                    Bill/Ship To:
                    Name:    John Doe
                    Address: [placeholder]
                    Email:   [placeholder]

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
                    - Use `[placeholder]` for missing customer information.  
                    - Maintain a **clear, professional tone**.  

                    ### Example Friendly Response Text:

                    "Thank you for your order, John Doe! We confirm your purchase of 1000 units of A4 Paper (80g/m²), scheduled for delivery on 2026-02-27. Please find your invoice below."  

                    Always ensure the invoice is accurate, well-formatted, and professional.
                """,
    tools=[],
)


class WorkflowContext(BaseModel):
    """Shared context between agents"""

    request_id: str
    request_body: str


class MultiAgentWorkflow:
    """
    Orchestrates the end-to-end processing of a customer request.

    The workflow follows these steps:

    1. Classify the request (INQUIRY / ORDER) via the orchestration agent.
    2. For INQUIRIES: query inventory, generate a quote and return it.
    3. For ORDERS: check stock, possibly reorder, generate a quote,
        finalize the sale, and produce an invoice.

    The class keeps a usage counter for each worker agent to aid debugging
    and reporting.
    """

    def __init__(self):
        """
        Initialize all agents and set up a usage counter.

        Agents are stored in `self.agents` for easy lookup.  A separate dictionary,
        `self.agent_usage_count`, tracks how many times each worker agent has been invoked.
        This information can be used for monitoring or billing purposes.
        """
        self.agents = {
            "orchestration": orchestration_agent,
            "inventory": inventory_agent,
            "quoting": quoting_agent,
            "ordering": ordering_agent,
            "invoice": invoice_agent,
        }

        self.agent_usage_count = {
            "orchestration": 0,
            "inventory": 0,
            "quoting": 0,
            "ordering": 0,
            "invoice": 0,
        }

    def handle_inquiry(self, context: WorkflowContext) -> str:
        """
        For an INQUIRY request we first ask the Inventory Agent for stock info,
        then pass that context to the Quoting Agent so it can produce a quote.
        """
        # 1. Ask inventory (even if no order)
        try:
            inv_resp = self.agents["inventory"].run_sync(
                f"""
                Classification: INQUIRY
                User Request: {context.request_body}
                """,
                deps=context,
                usage_limits=UsageLimits(request_limit=50),
            )
            self.agent_usage_count["inventory"] += 1
        except UsageLimitExceeded as e:
            logger.error(f"Inventory agent error on inquiry: {e}")
            return "We're sorry, but we couldn't check inventory at this time."

        # 2. Pass the inventory answer to quoting
        quote_prompt = f"""
            User Request: {context.request_body}
            Inventory Context: {inv_resp.output.answer}
        """
        try:
            quote_resp = self.agents["quoting"].run_sync(
                quote_prompt,
                deps=context,
                usage_limits=UsageLimits(request_limit=50),
            )
            self.agent_usage_count["quoting"] += 1
        except UsageLimitExceeded as e:
            logger.error(f"Quoting agent error on inquiry: {e}")
            return "We're sorry, but we couldn't generate a quote at this time."

        # Return the quoting output directly (plain text)
        return quote_resp.output

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
                deps=context,
                usage_limits=UsageLimits(request_limit=200),
            )
            self.agent_usage_count["inventory"] += 1
        except UsageLimitExceeded as e:
            logger.error(f"Usage Limit exceeded calling inventory agent: {e}")
            return "We're sorry, but our inventory cannot be verified at this time."

        # If the inventory agent says we cannot proceed now, stop here
        if inventory_response.output.proceed is False:
            reason = getattr(inventory_response.output, "rationale", "")
            return f"Cannot fulfill order at this time. Justification: {reason}"

        quote_prompt = f"""
            User Request: {context.request_body}
            Inventory Context: {inventory_response.output.answer}
        """

        # Call quoting agent to generate a quote based on the order
        try:
            quoting_response = self.agents["quoting"].run_sync(
                quote_prompt,
                deps=context,
                usage_limits=UsageLimits(request_limit=50),
            )
            self.agent_usage_count["quoting"] += 1
        except UsageLimitExceeded as e:
            logger.error(f"Usage Limit exceeded calling quoting agent: {e}")
            return "We're sorry, but we cannot provide a Quote at this time."

        sales_prompt = f"""
            User Request: {context.request_body}
            Inventory Context: {inventory_response.output.answer}
            Quote Context: {quoting_response.output}
        """

        # Call sales agent to finalize the order
        try:
            sales_response = self.agents["ordering"].run_sync(
                sales_prompt,
                deps=context,
                usage_limits=UsageLimits(request_limit=50),
            )
            self.agent_usage_count["ordering"] += 1
        except UsageLimitExceeded as e:
            logger.error(f"Usage Limit exceeded calling sales agent: {e}")
            return "We're sorry, but we cannot complete your Order at this time."

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
                deps=context,
                usage_limits=UsageLimits(request_limit=50),
            )
            self.agent_usage_count["invoice"] += 1
            return invoice_response.output
        except UsageLimitExceeded as e:
            logger.error(f"Usage Limit exceeded calling invoice agent: {e}")
            return "We're sorry, but we cannot generate your Invoice at this time."

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
            request_body=customer_request,
        )

        # Step 1: Call the orchestration agent to classify the request
        try:
            orchestration_response = self.agents["orchestration"].run_sync(
                customer_request,
                deps=self.workflow_context,
                usage_limits=UsageLimits(request_limit=50),
            )
            self.agent_usage_count["orchestration"] += 1
        except UsageLimitExceeded as e:
            logger.error(f"Usage Limit exceeded calling orchestration agent: {e}")
            return "We're sorry, but we're unable to correctly classify your request at this time."

        print(
            f"Orchestration Agent classified request as: {orchestration_response.output.classification}"
        )

        # Step 2: Based on classification, route to appropriate agents
        if orchestration_response.output.classification == "INQUIRY":
            # Handle inquiries with quoting and evaluation agents
            response = self.handle_inquiry(self.workflow_context)
        elif orchestration_response.output.classification == "ORDER":
            # Handle orders with inventory and sales agents
            response = self.handle_order(self.workflow_context)
        else:
            response = "Invalid classification received from orchestration agent."

        return str(response)  # guarantees a string


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
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="raise"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        logger.error(f"FATAL: Error loading test data: {e}")
        return

    # Read the CSV file again for consistency
    quote_requests_sample = pd.read_csv("quote_requests_sample.csv")

    # Convert 'request_date' to datetime and sort by this column
    quote_requests_sample["request_date"] = pd.to_datetime(
        quote_requests_sample["request_date"], format="%m/%d/%y"
    )
    quote_requests_sample = quote_requests_sample.sort_values("request_date")

    # Determine the initial date from the earliest request_date in the sample data
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")

    # Generate a financial report using the initial date and store it in `financial_report`
    financial_report = generate_financial_report(initial_date)

    # Extract current cash balance and inventory value from the generated report
    cash_balance = financial_report["cash_balance"]
    inventory_value = financial_report["inventory_value"]

    ###########################################
    # INITIALIZE YOUR MULTI AGENT SYSTEM HERE #
    ###########################################

    agent_workflow = MultiAgentWorkflow()

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print("=" * 80)
        print(f"===== Request {idx + 1} =====")  # type: ignore
        print("=" * 80)
        print(f"Context         : {row['job']} organizing {row['event']}")
        print(f"Request Date    : {request_date}")
        print(f"Cash Balance    : ${cash_balance:.2f}")
        print(f"Inventory Value : ${inventory_value:.2f}")

        # Process request
        full_request = f"{row['request']} (Requested on: {request_date})"

        #####################################################
        # USE YOUR MULTI AGENT SYSTEM TO HANDLE THE REQUEST #
        #####################################################

        # Runs the multi-agent workflow
        response = agent_workflow.run(full_request)

        # After the invoice agent records a sale, recompute the report using request date:
        financial_report = generate_financial_report(request_date)
        cash_balance = financial_report["cash_balance"]
        inventory_value = financial_report["inventory_value"]

        print(" " * 80)
        print(f"Response:          {response}")
        print(" " * 80)
        print(f"Cash Balance:      ${cash_balance:.2f}")
        print(f"Inventory Value:   ${inventory_value:.2f}")
        print(" " * 80)

        # Determine fulfillment status
        if isinstance(response, str) and "INVOICE" in response.upper():
            status = "fulfilled"
        else:
            status = "unfulfilled"

        results.append(
            {
                "request_id": idx + 1,  # type: ignore
                "request_date": request_date,
                "cash_balance": cash_balance,
                "inventory_value": inventory_value,
                "status": status,
                "response": response,
            }
        )

        # Sleeps for 1s
        time.sleep(1)

    ###################################
    # Final report                    #
    ###################################
    final_report = generate_financial_report(
        quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    )

    print("=" * 80)
    print(
        "============================ FINAL FINANCIAL REPORT ============================"
    )
    print("=" * 80)
    print(f"Final Cash:      ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    ###################################
    # Print the agents' calls summary #
    ###################################
    print("=" * 80)
    print(
        "============================== AGENT CALL SUMMARY =============================="
    )
    print("=" * 80)
    for agent, count in agent_workflow.agent_usage_count.items():
        print(f"{agent.capitalize()} agent: {count} times")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results


if __name__ == "__main__":
    results = run_test_scenarios()
