"""Test-double backend package for the vendored frontend tests.

This package is a **fixture**, not the ATTUNE transport. Three of the tests in
``apps/attune-ui/tests/`` spawn a Python interpreter and feed version-1 packets
through the browser-side validators; the teammate's tests import their own
``backend`` package for that, and that package is not part of ``frontend/``.

It lives at ``apps/backend`` because the tests run with ``cwd`` set two levels
above ``apps/attune-ui/tests``, which is where ``import backend`` resolves.

``src/nova2026/transport`` is the implementation of record. Nothing in the demo
imports this package; when the real transport lands, it replaces this fixture,
and a disagreement between them means this fixture is wrong.
"""
