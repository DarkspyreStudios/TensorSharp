Review these three independent services for a production launch. They were written by different teams and share no database or deployment process. I need a substantive correctness review with concrete execution interleavings, fixes, and deterministic regression tests for each service. Focus on duplicate requests, concurrent requests, and a process crash at any await. State the most serious defect in each service, cite the exact operation order that causes it, and recommend an implementable correction. Do not spend space on style. Keep the final review concise (one short section per service).

Assume all database calls below succeed unless the process crashes. Database isolation is READ COMMITTED. There are no triggers, implicit locks, hidden uniqueness constraints, or background reconciliation workers. HTTP retries can reach a different application process. Each code block is pseudocode, and its statements execute in the order shown. Authentication and input validation before these methods are correct. The current automated tests invoke each method sequentially with successful dependencies.

Service A: account recovery

The recovery service stores one row per cryptographically random reset token. The token column is a primary key; each row has UserId, ExpiresAt, and Used. The product contract says a token can change its account password exactly once, even if two password-reset submissions arrive simultaneously. Users may deliberately submit different new passwords using the same token. Password hashing is deterministic for this exercise, and its CPU cost can yield control between calls. A consumed token must remain consumed; the service must not disclose whether a supplied email address exists.

```csharp
async Task ResetPassword(string token, string newPassword) {
    var row = await db.SingleAsync(
        "SELECT UserId, ExpiresAt, Used FROM ResetTokens WHERE Token = @token", token);
    if (row.Used || row.ExpiresAt <= clock.UtcNow) throw InvalidToken();
    string hash = HashPassword(newPassword);
    await db.ExecuteAsync("UPDATE Users SET PasswordHash = @hash WHERE Id = @user",
        hash, row.UserId);
    await db.ExecuteAsync("UPDATE ResetTokens SET Used = 1 WHERE Token = @token", token);
    await audit.WriteAsync("password_reset_completed", row.UserId);
}
```

Each ExecuteAsync commits its own transaction. The team proposes fixing occasional duplicate completions by adding a C# lock around the method, but the service runs four processes behind a load balancer. Explain whether that proposed fix satisfies the contract, and identify a database-level change that makes the state transition and password write safe together.

Service B: recurring billing

The billing endpoint accepts a caller-generated idempotency key scoped to an account. BillingRequests has a composite primary key (AccountId, Key), and stores provider ChargeId, Amount, and ResponseJson. The external card provider can charge successfully even if the application crashes before the next line. The provider offers durable idempotent charging when the application supplies a stable provider idempotency key; the provider request below currently does not supply one. The contract is that retrying the same account/key/amount never creates a second external charge, and reusing the same account/key with a different amount must fail rather than silently reuse another payment's result.

```csharp
async Task<Receipt> Charge(string accountId, string key, decimal amount) {
    var previous = await db.FindAsync<BillingRequest>(accountId, key);
    if (previous != null) return DeserializeReceipt(previous.ResponseJson);
    var paid = await provider.ChargeAsync(accountId, amount); // no idempotency key
    var response = new Receipt(paid.ChargeId, amount);
    await db.InsertAsync(new BillingRequest(accountId, key, paid.ChargeId,
        amount, Serialize(response))); // uniqueness enforced here
    return response;
}
```

The billing team argues that the primary key prevents duplicate payments because a duplicate insert will fail. Evaluate that claim separately for concurrent first submissions and for a crash after provider success. Address how a retry learns the existing outcome without issuing another charge, and how the implementation should detect a changed amount. Do not assume a database transaction can roll back the remote card provider.

Service C: inventory reservations

Inventory has primary key Sku and integer Available. Reservations has primary key ReservationId and fields Sku and Quantity. Available is initially 5 for SKU red-widget. Two distinct customers can each request quantity 4 concurrently; only one may succeed. Quantities have already been validated positive. A successful response promises that a reservation row exists and stock was decremented exactly once. A failed database transaction must leave both tables unchanged.

```csharp
async Task<string> Reserve(string sku, int quantity) {
    var item = await db.FindAsync<Inventory>(sku);
    if (item.Available < quantity) throw OutOfStock();
    string reservationId = NewId();
    await db.ExecuteAsync("UPDATE Inventory SET Available = @remaining WHERE Sku = @sku",
        item.Available - quantity, sku);
    await db.InsertAsync(new Reservation(reservationId, sku, quantity));
    return reservationId;
}
```

Each write commits separately. The inventory team proposes retrying the method when an UPDATE throws, without changing the SQL or checking affected rows. Explain the observable outcome when both readers see Available=5, and the separate outcome if the process stops after stock is changed but before inserting the reservation. Give a transactional correction that checks available stock atomically and define a barrier-controlled test for two simultaneous reservations.
