# ActServe interactive architecture

This map explains the v0.10 runtime from a perishable robot observation to an
explicit request outcome. It is grounded in the public source tree at the
revision declared in
[`actserve-v0.10.architecture.json`](actserve-v0.10.architecture.json); it does
not claim closed-loop task-quality gains or hardware safety.

## Open the map

Download [`actserve-v0.10.html`](actserve-v0.10.html) and open it in a modern
browser, or serve the repository root locally:

```bash
python3 -m http.server 8000
```

Then visit
`http://127.0.0.1:8000/docs/architecture/actserve-v0.10.html`.

The standalone page includes:

- four guided chapters for the observation, backend, result, and optional
  action-queue paths;
- source links pinned to a public Git revision;
- finite live trace motion, node search, route and reach exploration;
- light/dark themes plus PNG, SVG, WebM, and share-card export.

## Update with Archify

The JSON file is the editable source of truth. Archify's preview command watches
it and refreshes only after the new revision passes validation, so a broken
half-edit does not replace the last good diagram.

```bash
git clone --depth 1 --branch v2.13.0 \
  https://github.com/tt-a1i/archify.git /tmp/archify

node /tmp/archify/archify/bin/archify.mjs preview architecture \
  docs/architecture/actserve-v0.10.architecture.json \
  docs/architecture/actserve-v0.10.html \
  --quality showcase \
  --repo-root .
```

When the implementation changes, update both the topology and
`meta.repository.revision`, validate the source references, then commit the JSON,
HTML, and exported share card together. Archify provides a validated live
authoring loop; it does not infer architecture changes from code automatically.

Generated with [Archify v2.13.0](https://github.com/tt-a1i/archify), released
under the MIT License.
