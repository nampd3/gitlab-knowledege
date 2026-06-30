# Static assets

This directory is served at `/static/<filename>` by the
`Visualization_Server` for any file you drop here.

## Currently installed

| File | Size | Role |
|---|---|---|
| `cytoscape.min.js` | ~400 KB | Interactive renderer for the `/dependencies` knowledge graph. Zoom, pan, drag, force-directed layout. |
| `mermaid.min.js` | ~3.5 MB | Documented fallback. Used when `cytoscape.min.js` is not present. |

## How rendering works

The `/dependencies` page emits the diagram twice:

* As a Cytoscape-shaped JSON payload inside
  `<script type="application/json" class="graph-data">...</script>`.
* As Mermaid source inside `<pre class="mermaid">graph LR ...</pre>`
  (kept for spec compatibility and as the fallback).

The inline page chrome script then:

1. Checks for the global `cytoscape`. If present, it parses every
   `.graph-data` block, hides the matching `<pre class="mermaid">`
   via CSS, and renders the JSON into the adjacent
   `<div class="graph-container">` with a force-directed
   (`cose`) layout. This is what an operator with the bundle
   installed sees: an interactive knowledge graph with zoom, pan,
   drag, and tooltips.
2. If `cytoscape` is undefined (the file is missing or failed to
   load), the script adds `body.no-cytoscape`, which inverts the
   default CSS rules: the Mermaid `<pre>` block becomes visible,
   the empty Cytoscape container is hidden, and `mermaid.initialize`
   runs with the documented size and layout overrides.

No-JavaScript mode falls back further still: the page shows the raw
`graph LR ...` source text inside the `<pre>` block.

## Cytoscape — primary renderer

### Install

```bash
curl -L -o cytoscape.min.js \
  https://cdn.jsdelivr.net/npm/cytoscape@3/dist/cytoscape.min.js
```

Place the file at
`src/project_knowledge_mcp/static/cytoscape.min.js` so the running
server serves it at `http://127.0.0.1:7345/static/cytoscape.min.js`.

### Verifying the install

```bash
curl -s -o /dev/null -w '%{http_code}  bytes=%{size_download}  ct=%{content_type}\n' \
  http://127.0.0.1:7345/static/cytoscape.min.js
# Expected: 200  bytes=~400000  ct=text/javascript; charset=utf-8
```

Reload `/dependencies`. The page should show an
interactive node-and-edge graph with `<div class="graph-container">`
as its drawing surface. Use the mouse wheel to zoom, click-drag
empty space to pan, and click-drag a node to reposition it. Edge
labels appear on hover.

### Refreshing the bundle

Replace `cytoscape.min.js` with a newer release; no server restart
needed (the Starlette static mount reads files per request). Any
Cytoscape 3.x release will work — the page only calls
`cytoscape({ container, elements, layout, style })` with no
extension dependencies.

## Mermaid — fallback renderer

Kept on disk so an air-gapped operator without Cytoscape installed
still sees a rendered diagram (the previously-supported behavior).
Install and refresh instructions are identical to Cytoscape's:

```bash
curl -L -o mermaid.min.js \
  https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js
```

Smoke test:

```bash
curl -s -o /dev/null -w '%{http_code}  bytes=%{size_download}  ct=%{content_type}\n' \
  http://127.0.0.1:7345/static/mermaid.min.js
# Expected: 200  bytes=3565102  ct=text/javascript; charset=utf-8
```

The fallback is activated automatically by the inline page chrome
script when `cytoscape` is undefined. No code or configuration
change is required.

### Air-gapped / restricted networks

If the host running the server cannot reach `cdn.jsdelivr.net`, fetch
the file on a workstation that can and copy it in via scp / rsync /
artifact server / USB. Both Cytoscape 3.x and Mermaid 10.x or 11.x
work with the existing page chrome.

## Removing either renderer

Delete the file. The chrome's `<script src="/static/<...>">` tag
still gets emitted, the browser receives HTTP 404, the corresponding
global stays undefined, and the inline script's `typeof` guard
takes the documented degraded path (Mermaid fallback when Cytoscape
is absent; raw text when both are absent).

## Adding other static files

Any file dropped into this directory is served at
`/static/<filename>` automatically. The mount uses Starlette's
`StaticFiles` with `check_dir=False`, so an empty directory does not
prevent the server from starting; a missing file returns 404 and
falls through to the standard 404 fallback handler.

Path-traversal attempts (`/static/../../secret`) are rejected by
Starlette with a 404 — the mount cannot reach outside this directory.
