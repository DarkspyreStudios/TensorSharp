# Inventory service review

A reservation is released when payment fails or its lease expires. Updates use a
version check to prevent lost writes under concurrent reservations.

FACT: inventory.reservation_minutes=12
FACT: inventory.version_conflict_retries=4
FACT: inventory.low_stock_threshold=9

An earlier prototype used a ten-minute lease and two retries. Those are not the
current values. Report the FACT lines above.
