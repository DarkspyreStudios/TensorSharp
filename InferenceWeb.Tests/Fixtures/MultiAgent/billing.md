# Billing service review

Retries preserve the external idempotency key. A settlement event can be delivered
more than once; a uniqueness constraint makes handling safe.

FACT: billing.idempotency_retention_hours=72
FACT: billing.retry_delays_seconds=2,8,32
FACT: billing.currency_rounding=half_even

The archive mentions a twenty-four-hour retention window and half-up rounding.
Those are obsolete. Report the current FACT lines above.
