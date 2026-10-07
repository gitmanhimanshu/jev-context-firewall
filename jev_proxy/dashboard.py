import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class LogEntry:
    timestamp: str
    path: str
    original_tokens: int
    pruned_tokens: int
    reduction_pct: float
    latency_ms: int
    status: str

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "path": self.path,
            "original_tokens": self.original_tokens,
            "pruned_tokens": self.pruned_tokens,
            "reduction_pct": round(self.reduction_pct, 1),
            "latency_ms": self.latency_ms,
            "status": self.status,
        }


@dataclass
class SessionMetrics:
    total_requests: int = 0
    total_original_tokens: int = 0
    total_pruned_tokens: int = 0
    total_tokens_saved: int = 0
    average_reduction_pct: float = 0.0
    estimated_cost_saved: float = 0.0
    recent_logs: List[LogEntry] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "total_original_tokens": self.total_original_tokens,
            "total_pruned_tokens": self.total_pruned_tokens,
            "total_tokens_saved": self.total_tokens_saved,
            "average_reduction_pct": round(self.average_reduction_pct, 1),
            "estimated_cost_saved": round(self.estimated_cost_saved, 3),
            "recent_logs": [log.to_dict() for log in self.recent_logs],
        }


class Tracker:
    def __init__(self):
        self.metrics = SessionMetrics()
        self._lock = asyncio.Lock()

    async def record_request(
        self,
        path: str,
        original_tokens: int,
        pruned_tokens: int,
        latency_ms: int,
        status: str,
    ) -> None:
        async with self._lock:
            saved = max(0, original_tokens - pruned_tokens)
            self.metrics.total_requests += 1
            self.metrics.total_original_tokens += original_tokens
            self.metrics.total_pruned_tokens += pruned_tokens
            self.metrics.total_tokens_saved += saved

            if self.metrics.total_original_tokens > 0:
                self.metrics.average_reduction_pct = (
                    float(self.metrics.total_tokens_saved)
                    / float(self.metrics.total_original_tokens)
                    * 100.0
                )

            # Assuming average $3.00 per 1M tokens saved
            self.metrics.estimated_cost_saved = (
                float(self.metrics.total_tokens_saved) / 1000000.0
            ) * 3.00

            red_pct = 0.0
            if original_tokens > 0:
                red_pct = (float(saved) / float(original_tokens)) * 100.0

            entry = LogEntry(
                timestamp=datetime.now().strftime("%H:%M:%S"),
                path=path,
                original_tokens=original_tokens,
                pruned_tokens=pruned_tokens,
                reduction_pct=red_pct,
                latency_ms=latency_ms,
                status=status,
            )

            self.metrics.recent_logs.insert(0, entry)
            if len(self.metrics.recent_logs) > 30:
                self.metrics.recent_logs = self.metrics.recent_logs[:30]

    def get_metrics_dict(self) -> dict:
        return self.metrics.to_dict()


global_tracker = Tracker()

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>🛡️ Jev Context Firewall Dashboard (Python)</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    :root {
      --bg: #090d16;
      --card-bg: rgba(22, 28, 45, 0.85);
      --border: rgba(255, 255, 255, 0.08);
      --accent: #3b82f6;
      --green: #10b981;
      --fg: #f3f4f6;
      --muted: #9ca3af;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background: var(--bg); color: var(--fg); min-height: 100vh; padding: 2rem; }
    .container { max-width: 1100px; margin: 0 auto; }
    header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem; }
    h1 { font-size: 1.5rem; display: flex; align-items: center; gap: 0.6rem; }
    .badge { background: rgba(16, 185, 129, 0.15); color: var(--green); padding: 0.25rem 0.75rem; border-radius: 999px; font-size: 0.8rem; border: 1px solid rgba(16, 185, 129, 0.3); }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; margin-bottom: 2rem; }
    .card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 1.25rem; backdrop-filter: blur(10px); }
    .card-title { color: var(--muted); font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem; }
    .card-value { font-size: 1.8rem; font-weight: 700; color: #fff; }
    .card-value.highlight { color: var(--green); text-shadow: 0 0 15px rgba(16, 185, 129, 0.4); }
    .table-card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }
    table { width: 100%; border-collapse: collapse; text-align: left; font-size: 0.9rem; }
    th { background: rgba(255, 255, 255, 0.03); color: var(--muted); padding: 0.85rem 1.25rem; border-bottom: 1px solid var(--border); font-weight: 600; }
    td { padding: 0.85rem 1.25rem; border-bottom: 1px solid var(--border); }
    tr:last-child td { border-bottom: none; }
    .pill { display: inline-block; padding: 0.2rem 0.5rem; border-radius: 6px; font-size: 0.75rem; font-weight: 600; }
    .pill.success { background: rgba(16, 185, 129, 0.15); color: var(--green); }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1><span>🛡️</span> Jev Context Firewall (Python) <span class="badge">Online</span></h1>
      <div style="display: flex; align-items: center; gap: 1rem;">
        <a href="/api/stats/export" download class="pill" style="text-decoration:none; background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.4); padding: 0.35rem 0.75rem; cursor:pointer;">📥 Export JSON Report</a>
        <span style="color: var(--muted); font-size: 0.85rem;" id="last-update">Updating...</span>
      </div>
    </header>

    <div class="grid">
      <div class="card">
        <div class="card-title">Total Requests</div>
        <div class="card-value" id="val-reqs">0</div>
      </div>
      <div class="card">
        <div class="card-title">Tokens Saved</div>
        <div class="card-value highlight" id="val-saved">0</div>
      </div>
      <div class="card">
        <div class="card-title">Average Reduction</div>
        <div class="card-value highlight" id="val-pct">0.0%</div>
      </div>
      <div class="card">
        <div class="card-title">Est. Cost Saved</div>
        <div class="card-value" id="val-cost">$0.00</div>
      </div>
    </div>

    <div class="table-card">
      <div style="padding: 1.25rem; border-bottom: 1px solid var(--border); font-weight: 600;">Recent Intercepted Agent Turns</div>
      <table>
        <thead>
          <tr>
            <th>Time</th>
            <th>Endpoint</th>
            <th>Original</th>
            <th>Pruned</th>
            <th>Reduction</th>
            <th>Latency</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody id="log-body">
          <tr><td colspan="7" style="text-align: center; color: var(--muted);">Waiting for first IDE request...</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <script>
    async function update() {
      try {
        const res = await fetch('/api/stats');
        const data = await res.json();
        document.getElementById('val-reqs').innerText = data.total_requests.toLocaleString();
        document.getElementById('val-saved').innerText = data.total_tokens_saved.toLocaleString();
        document.getElementById('val-pct').innerText = data.average_reduction_pct.toFixed(1) + '%';
        document.getElementById('val-cost').innerText = '$' + data.estimated_cost_saved.toFixed(3);
        document.getElementById('last-update').innerText = 'Last updated: ' + new Date().toLocaleTimeString();

        const tbody = document.getElementById('log-body');
        if (data.recent_logs && data.recent_logs.length > 0) {
          tbody.innerHTML = data.recent_logs.map(function(l) {
            return '<tr>' +
              '<td>' + l.timestamp + '</td>' +
              '<td style="font-family: monospace;">' + l.path + '</td>' +
              '<td>' + l.original_tokens.toLocaleString() + '</td>' +
              '<td>' + l.pruned_tokens.toLocaleString() + '</td>' +
              '<td style="color: #10b981; font-weight: 600;">' + l.reduction_pct.toFixed(1) + '%</td>' +
              '<td>' + l.latency_ms + 'ms</td>' +
              '<td><span class="pill success">' + l.status + '</span></td>' +
            '</tr>';
          }).join('');
        }
      } catch (err) {
        console.error(err);
      }
    }
    setInterval(update, 2000);
    update();
  </script>
</body>
</html>
"""
