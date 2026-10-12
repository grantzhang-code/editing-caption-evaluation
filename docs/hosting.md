# Metric guide hosting

The static guide lives in `site/index.html`; `site/summary.html` contains the
verified five-way aggregate table. Both work without external fonts or scripts.
The repository includes aggregate JSON/CSV in `data/`, but no case captions,
image assets or provider replies. The internal Highway viewer additionally
offers case-level inspection using private generated assets.

Live guide: https://grantzhang-alpha-document.highway.canva-internal.dev/

Highway route: `grantzhang-alpha-document`, with WARP access enabled.
The service runs on port `18981` in tmux session `editing-metrics-1008`,
with windows `web` and `highway`. The tmux processes persist after disconnecting
the shell; the Devbox and both processes must remain running.

To serve locally from the repository root:

```bash
python -m http.server 18982 --bind 127.0.0.1
```

Open `http://127.0.0.1:18982/site/` for the guide or
`http://127.0.0.1:18982/site/summary.html` for the table. Serving the repository
root also makes the linked `data/` downloads available.

The existing internal viewer continues to run on port 18981. To resume its
route after restarting that service:

```bash
infra highway run --route grantzhang-alpha-document --port 18981
```

Keep that command running in tmux. The browser print button expands details for
a PDF export. The Q and edge calculators mirror the Python formulas, including
critical-error gating, the IoU threshold, edit/keep behavior and unmatched-edge
denominators. They do not issue API calls.

## Update the aggregate table

Use a verified five-way viewer index:

```bash
python scripts/export_metric_summary.py --index /path/to/verified/index.json --output .
```

The exporter independently recomputes all 55 means and each metric's paired N,
then writes `data/five-way-metrics-20261009.json`, its CSV companion,
`docs/results.md` and `site/summary.html`. Only explicit aggregate fields are
exported, with the source-index SHA-256 for provenance.

To append separately verified consistency scores to an internal viewer index:

```bash
python scripts/append_consistency_results.py \
  --data /path/to/base-five-way-viewer \
  --final /path/to/verified-consistency/results/final \
  --output /path/to/new-viewer-directory
```

The append step checks the final manifest, source-index hash, caption/GT
bindings, protocol, F1 arithmetic and paired means. Its output contains private
case data and must stay outside the public Git tree.
