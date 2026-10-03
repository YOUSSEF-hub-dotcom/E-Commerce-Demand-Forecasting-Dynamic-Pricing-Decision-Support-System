"""Production ingestion, merge, and order-status policy for the Olist dataset.

Re-architects the Phase-1 merge logic (certified in notebook 01) as importable,
re-tested modules. All inputs are read from ``data/raw`` and never mutated.

Pipeline contract (enforced by ``tests/test_data_processing.py``):

1. ``load_core_tables`` - read the four core CSVs, parse every timestamp to
   ``datetime64``.
2. ``merge_core_tables`` - item-level fact table via the documented join
   graph: order_items + orders (order_id) + products (product_id) +
   category_translation (product_category_name). ``validate="many_to_one"``
   enforces key uniqueness on the right side of every hop.
3. ``apply_status_policy`` - restrict the demand stream to economically valid
   orders. Default rule: exclude ``canceled`` and ``unavailable`` only. Orders
   that were genuinely placed but simply not *yet* delivered (shipped,
   invoiced, processing, created, approved) are genuine demand and are kept.
   Filtering strictly to ``delivered`` would truncate the most recent weeks of
   the timeline (delivery latency), biasing demand artificially low at the end
   of the series.
"""

from pathlib import Path

import pandas as pd

from src.utils.logging_setup import get_logger

LOGGER = get_logger(__name__)

ORDER_DATETIME_COLS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

# Economically invalid order states - they never translated into shipped demand.
# Kept as a module constant so tests can assert the business rule explicitly.
EXCLUDED_ORDER_STATUSES = frozenset({"canceled", "unavailable"})


def load_core_tables(raw_dir: Path | str) -> dict[str, pd.DataFrame]:
    """Read the four core Olist CSVs with correct dtypes.

    Parameters
    ----------
    raw_dir : Path | str
        Directory containing the raw Olist ``.csv`` files (``data/raw``).

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of table name to DataFrame: ``orders``, ``items``,
        ``products``, ``translation``.

    Raises
    ------
    FileNotFoundError
        If any of the four required files is missing from ``raw_dir``.
    """
    raw_dir = Path(raw_dir)
    expected = [
        "olist_orders_dataset.csv",
        "olist_order_items_dataset.csv",
        "olist_products_dataset.csv",
        "product_category_name_translation.csv",
    ]
    for fname in expected:
        if not (raw_dir / fname).exists():
            raise FileNotFoundError(f"Missing raw dataset: {raw_dir / fname}")

    orders = pd.read_csv(raw_dir / "olist_orders_dataset.csv", parse_dates=ORDER_DATETIME_COLS)
    items = pd.read_csv(raw_dir / "olist_order_items_dataset.csv", parse_dates=["shipping_limit_date"])
    products = pd.read_csv(raw_dir / "olist_products_dataset.csv")
    translation = pd.read_csv(raw_dir / "product_category_name_translation.csv")

    for col in ORDER_DATETIME_COLS:
        if not pd.api.types.is_datetime64_any_dtype(orders[col]):
            raise TypeError(f"orders.{col} was not parsed to datetime64.")

    LOGGER.info("Loaded orders=%d items=%d products=%d translation=%d",
                len(orders), len(items), len(products), len(translation))
    return {"orders": orders, "items": items, "products": products, "translation": translation}


def merge_core_tables(
    items: pd.DataFrame,
    orders: pd.DataFrame,
    products: pd.DataFrame,
    translation: pd.DataFrame,
) -> pd.DataFrame:
    """Build the item-level fact table from the four core Olist tables.

    Applies the certified left-join graph and logs the row budget after each
    hop. ``validate="many_to_one"`` is used to fail loudly if a right-side key
    is ever duplicated, which would silently explode the row count.

    Parameters
    ----------
    items : pd.DataFrame
        Order-items table (left side, 1 row per sold item).
    orders : pd.DataFrame
        Orders table.
    products : pd.DataFrame
        Products metadata table.
    translation : pd.DataFrame
        Portuguese->English category translation table.

    Returns
    -------
    pd.DataFrame
        Fact table with 1 row per item.
    """
    LOGGER.info("merge_core_tables: items=%d, orders=%d", len(items), len(orders))

    df = items.merge(orders, on="order_id", how="left", validate="many_to_one")
    LOGGER.info("  +orders  -> %d x %d", *df.shape)
    df = df.merge(products, on="product_id", how="left", validate="many_to_one")
    LOGGER.info("  +products -> %d x %d", *df.shape)
    df = df.merge(translation, on="product_category_name", how="left", validate="many_to_one")
    LOGGER.info("  +translation -> %d x %d", *df.shape)

    if df["order_purchase_timestamp"].isna().any():
        orphans = int(df["order_purchase_timestamp"].isna().sum())
        raise ValueError(f"{orphans} items failed to resolve their order_id.")
    return df


def apply_status_policy(fact: pd.DataFrame) -> pd.DataFrame:
    """Filter the fact table to economically valid order statuses.

    Parameters
    ----------
    fact : pd.DataFrame
        Merged fact table (must contain ``order_status``).

    Returns
    -------
    pd.DataFrame
        Fact table with invalid statuses removed.

    Raises
    ------
    ValueError
        If ``order_status`` is absent from the input schema.
    """
    if "order_status" not in fact.columns:
        raise ValueError("apply_status_policy: fact table is missing order_status column.")

    excluded = sorted(EXCLUDED_ORDER_STATUSES)
    dropped = fact["order_status"].isin(excluded)
    filtered = fact[~dropped].copy()
    LOGGER.info(
        "apply_status_policy: dropped %d invalid rows (%s) -> %d rows remain",
        int(dropped.sum()), excluded, len(filtered),
    )
    return filtered


def build_fact_table(raw_dir: Path | str) -> pd.DataFrame:
    """End-to-end fact-table construction from raw files.

    One-call convenience that chains ``load_core_tables``,
    ``merge_core_tables`` and ``apply_status_policy``. Used by notebooks and
    integration tests.

    Parameters
    ----------
    raw_dir : Path | str
        Directory containing the raw Olist ``.csv`` files.

    Returns
    -------
    pd.DataFrame
        Valid-order item-level fact table.
    """
    tables = load_core_tables(raw_dir)
    fact = merge_core_tables(**tables)
    return apply_status_policy(fact)