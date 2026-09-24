# Authentication service review

Password reset creates a single-use token; consumption and revocation occur in one
transaction. The endpoint uses the configured expiry rather than the session TTL.

FACT: auth.reset_token_minutes=15
FACT: auth.failed_attempt_limit=7
FACT: auth.session_cookie=HttpOnly,Secure,SameSite=Lax

The old migration used five attempts and a thirty-minute expiry. Those historical
values do not describe the current implementation. Report the FACT lines above.
