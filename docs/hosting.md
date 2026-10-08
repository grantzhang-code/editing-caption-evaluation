# Metric guide hosting

The static guide lives in `site/`; it has no external font, script or data
dependencies. It contains only metric explanations and synthetic calculator
examples, with no case records, credentials or image assets.

Live guide: https://grantzhang-alpha-document.highway.canva-internal.dev/

Highway route: `grantzhang-alpha-document`, with WARP access enabled.
The service runs on port `18981` in tmux session `editing-metrics-1008`,
with windows `web` and `highway`. The tmux processes persist after disconnecting
the shell; the Devbox and both processes must remain running.

To serve locally from the repository root:

```bash
python -m http.server 18981 --bind 127.0.0.1 --directory site
```

To resume the existing route after restarting the HTTP server:

```bash
infra highway run --route grantzhang-alpha-document --port 18981
```

Keep that command running in tmux. The browser print button expands details for
a PDF export. The Q and edge calculators mirror the Python formulas, including
critical-error gating, the IoU threshold, edit/keep behavior and unmatched-edge
denominators. They do not issue API calls.
