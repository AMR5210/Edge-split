# EdgeSplit dashboard

The dashboard is a local, presentation-only control surface for the existing
EdgeSplit router. It is intentionally separate from V1/V2 orchestration.

```bash
cd ~/edgesplit
./dashboard/start_dashboard.sh
```

Open `http://127.0.0.1:8084` in a browser running on the laptop. The service
is loopback-only and never starts, stops, installs, or rebuilds anything.

The page proxies explicit V1/V2 generation requests to the existing router at
`http://127.0.0.1:8083`. Start the existing laptop demo/router and the phone
services manually first. Phone readiness remains manual; this dashboard only
polls the existing phone decode and V1-receiver health routes.

The two terminal-styled panes are read-only activity mirrors, not shell
embeds. They show dashboard proxy events and existing health observations; the
raw V2 TCP listener is deliberately not sent an HTTP health probe.
