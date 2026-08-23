"""Order lookup tool: customer-safe order status retrieval.

Pure Python, no LLM/agent logic, no network calls. Customer-unsafe data
(customer contact details; the whole `internal` block) is structurally
excluded from the result models -- a leak would require first adding a field
to hold it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "orders.json"

_NON_ALNUM_RE = re.compile(r"[^A-Z0-9]+")
_DIGITS_ONLY_RE = re.compile(r"[0-9]+")


class OrderLookupItem(BaseModel):
    name: str
    quantity: int
    final_sale: bool


class OrderLookupResult(BaseModel):
    """Customer-safe view of one order.

    Fields for customer.name/email/shipping_address and everything under
    `internal` (risk_score, warehouse_note, support_tags) intentionally DO
    NOT EXIST on this model; lookup_order() populates it via an explicit
    whitelist, so sensitive values are structurally impossible to include.
    """

    found: bool = False
    order_id: str | None = None
    membership_tier: str | None = None
    items: list[OrderLookupItem] = []
    placed_at: str | None = None
    status: str | None = None
    status_updated_at: str | None = None
    shipped_at: str | None = None
    delivered_at: str | None = None
    carrier: str | None = None
    tracking_number: str | None = None
    estimated_delivery: str | None = None
    customer_safe_message: str | None = None
    status_note: str | None = None
    handoff_recommended: bool = False
    handoff_reason: str | None = None


@dataclass
class OrderCatalog:
    """Orders indexed by normalized ID plus the dataset snapshot timestamp."""

    by_id: dict[str, dict[str, Any]]
    snapshot_at: str | None


def normalize_order_id(raw: str) -> str:
    """Uppercase, trim whitespace, drop ALL non-alphanumerics.

    Applied identically to stored keys (load_orders) and user input, so
    "ord-1004", "(ORD_1004)", "  ord 1004  " all match ORD-1004. Anything
    that still does not match exactly falls through to clean not-found --
    never fuzzy-matched to a different real order.
    """
    return _NON_ALNUM_RE.sub("", raw.strip().upper())


def load_orders(path: str | Path) -> OrderCatalog:
    """Load orders.json once; index by normalized order_id for O(1) lookup."""
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    catalog = {
        normalize_order_id(record["order_id"]): record
        for record in payload.get("orders", [])
    }
    return OrderCatalog(by_id=catalog, snapshot_at=payload.get("snapshot_at"))


def lookup_order(
    order_id: str,
    orders_by_id: dict[str, dict[str, Any]],
    snapshot_at: str = "",
) -> OrderLookupResult:
    """Look up one order and build the customer-safe result.

    Status precedence rules enforced HERE, unambiguously:
    * cancelled/returned -> carrier/tracking_number/estimated_delivery are
      NULLED in the result itself (they are historical artifacts and must not
      be presented as current transit info). shipped_at/delivered_at describe
      what actually happened and stay. The raw stale values may go to
      human-facing trace/debug logging (Phase 6) but never re-enter model
      context.
    * shipped with null estimated_delivery -> status_note says an estimate is
      unavailable; nothing is ever calculated or invented.
    * exception -> handoff_recommended=True for human review.
    * unknown/blank after normalization -> found=False, all other fields at
      their defaults; nothing fabricated.
    * near-miss bare numeric IDs ("1007") -> if the canonical "ORD"+digits
      form names a real order, that order is returned; otherwise clean
      not-found. Deterministic tool-layer recovery -- never fuzzy-matched,
      and inputs containing letters never enter this branch.

    snapshot_at is accepted for interface stability but unused today -- the
    dataset bakes authoritative truth into status/customer_safe_message. It
    matters later (e.g. Phase 8's 30-minute cancellation-window logic).
    """
    del snapshot_at

    normalized = normalize_order_id(order_id)
    record = orders_by_id.get(normalized) if normalized else None
    if record is None and normalized and _DIGITS_ONLY_RE.fullmatch(normalized):
        # Near-miss recovery: customers often type "order 1007" without the
        # ORD prefix. Keys here are ALREADY normalized (hyphens stripped),
        # so the canonical candidate is "ORD" + digits. Accept it only if
        # that exact order exists; otherwise fall through to clean
        # not-found. Alphanumeric garbage ("ASDF1234") never reaches this
        # branch, preserving strict garbage-ID handling.
        record = orders_by_id.get("ORD" + normalized)
    if record is None:
        return OrderLookupResult(found=False)

    status = record.get("status")
    closed = status in ("cancelled", "returned")

    if closed:
        note = (
            "This order is cancelled and will not be shipped."
            if status == "cancelled"
            else "This order has been returned and is closed."
        )
    elif status == "shipped" and not record.get("estimated_delivery"):
        note = (
            "The order has shipped; a delivery estimate is currently "
            "unavailable."
        )
    else:
        note = None

    handoff_recommended = False
    handoff_reason: str | None = None
    if status == "exception":
        handoff_recommended = True
        handoff_reason = (
            "The shipment has an exception that requires support review."
        )

    return OrderLookupResult(
        found=True,
        order_id=record["order_id"],
        membership_tier=record.get("membership_tier"),
        items=[
            OrderLookupItem(
                name=item["name"],
                quantity=item["quantity"],
                final_sale=item["final_sale"],
            )
            for item in record.get("items", [])
        ],
        placed_at=record.get("placed_at"),
        status=status,
        status_updated_at=record.get("status_updated_at"),
        shipped_at=record.get("shipped_at"),
        delivered_at=record.get("delivered_at"),
        carrier=None if closed else record.get("carrier"),
        tracking_number=None if closed else record.get("tracking_number"),
        estimated_delivery=None if closed else record.get("estimated_delivery"),
        customer_safe_message=record.get("customer_safe_message"),
        status_note=note,
        handoff_recommended=handoff_recommended,
        handoff_reason=handoff_reason,
    )


_TEST_LOOKUPS = [
    "ORD-1001",
    "ord-1007 ",
    "ORD-1004",
    "ORD-1010",
    "ORD-1011",
    "ORD-1008",
    "ORD-9999",
    "  ord-1002  ",
    "1007",
    "asdf1234",
]


def main() -> None:
    catalog = load_orders(_DATA_PATH)
    print(
        f"snapshot_at={catalog.snapshot_at}; "
        f"{len(catalog.by_id)} orders indexed\n"
    )
    for raw in _TEST_LOOKUPS:
        result = lookup_order(raw, catalog.by_id, catalog.snapshot_at or "")
        print(f"=== lookup({raw!r}) normalized={normalize_order_id(raw)!r}")
        print(result.model_dump_json(indent=2))
        print()


if __name__ == "__main__":
    main()
