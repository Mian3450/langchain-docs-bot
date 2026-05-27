"""Use the operating system's trust store for TLS verification.

In corporate environments an intercepting proxy presents certificates signed by
a private root CA that is installed in the OS trust store but absent from the
``certifi`` bundle that Python/httpx use by default — causing
``CERTIFICATE_VERIFY_FAILED``. ``truststore`` makes Python's ``ssl`` module use
the OS trust store instead, which transparently fixes this while remaining a
no-op on machines without a custom CA.

Call :func:`enable_system_trust_store` once at process start, before any HTTPS
request (GitHub fetch, OpenAI API). It is best-effort: if ``truststore`` is not
installed it logs and returns, leaving the default behaviour intact.
"""

from __future__ import annotations

import structlog

log = structlog.get_logger(__name__)


def enable_system_trust_store() -> bool:
    """Route TLS verification through the OS trust store. Returns success."""
    try:
        import truststore
    except ImportError:
        log.debug("truststore_unavailable", note="using certifi default CA bundle")
        return False
    truststore.inject_into_ssl()
    log.debug("system_trust_store_enabled")
    return True
