"""Unit tests for ``diagram_renderer._encode_graph_data_for_html``.

The encoder turns a Cytoscape-shaped dict into a JSON string that is
safe to embed inside a ``<script type="application/json"
class="graph-data">...</script>`` block. Three guarantees are pinned
here:

1. **Valid JSON.** The output round-trips through :func:`json.loads`
   back into the original dict (modulo Unicode escapes for line and
   paragraph separators).
2. **``</`` is escaped.** The browser's HTML parser still treats a
   literal ``</script>`` as the end of the surrounding ``<script>``
   block regardless of ``type=``, so any ``</`` that would otherwise
   appear inside the JSON payload must be escaped to ``<\\/``.
3. **U+2028 / U+2029 are escaped.** JSON permits raw line- and
   paragraph-separator characters; the browser's JavaScript parser
   does not. Even though the JSON is consumed by ``JSON.parse``
   rather than evaluated as JS, escaping the two characters keeps
   the script block well-formed.
"""

from __future__ import annotations

import json

import pytest

from project_knowledge_mcp.diagram_renderer import _encode_graph_data_for_html

pytestmark = pytest.mark.unit


def test_encoder_produces_valid_json_for_empty_graph() -> None:
    """An empty graph encodes to JSON that round-trips cleanly."""
    encoded = _encode_graph_data_for_html({"nodes": [], "edges": []})

    assert json.loads(encoded) == {"nodes": [], "edges": []}


def test_encoder_round_trips_a_realistic_graph() -> None:
    """A non-trivial node + edge payload survives encode + JSON.parse."""
    data = {
        "nodes": [
            {"data": {"id": "P7", "label": "7 acme/auth"}},
            {"data": {"id": "P42", "label": "42 acme/payments"}},
        ],
        "edges": [
            {
                "data": {
                    "source": "P7",
                    "target": "P42",
                    "label": "shared table: orders",
                    "kind": "shared_table",
                }
            }
        ],
    }

    encoded = _encode_graph_data_for_html(data)

    assert json.loads(encoded) == data


def test_encoder_escapes_close_script_sequence() -> None:
    """A label containing ``</script>`` cannot close the surrounding tag.

    The escape rewrites every ``</`` as ``<\\/``. The result is still
    valid JSON (the backslash-slash escape is a JSON-permitted
    representation of ``/``) and the browser's HTML parser no longer
    sees a literal ``</script>`` sequence inside the payload.
    """
    data = {
        "nodes": [
            {"data": {"id": "P1", "label": "</script><script>alert(1)"}}
        ],
        "edges": [],
    }

    encoded = _encode_graph_data_for_html(data)

    assert "</script>" not in encoded
    assert "<\\/" in encoded
    # The decoded payload still carries the original label so
    # Cytoscape sees the intended (sanitized at render time) string.
    assert json.loads(encoded) == data


@pytest.mark.parametrize(
    ("raw", "expected_escape"),
    [
        ("\u2028", "\\u2028"),
        ("\u2029", "\\u2029"),
    ],
)
def test_encoder_escapes_line_and_paragraph_separators(
    raw: str, expected_escape: str
) -> None:
    """U+2028 and U+2029 are emitted as ``\\u202X`` to keep the script
    block valid.

    JSON tolerates the raw characters but the browser's JS tokenizer
    historically did not. Escaping is universally safe and keeps the
    encoded body printable.
    """
    data = {"nodes": [{"data": {"id": "P1", "label": f"line{raw}break"}}], "edges": []}

    encoded = _encode_graph_data_for_html(data)

    # The raw character must not appear; the escape must.
    assert raw not in encoded
    assert expected_escape in encoded
    # And the encoded payload still parses to the original dict.
    assert json.loads(encoded) == data
