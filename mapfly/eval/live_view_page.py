# HTML documents; line length is not a useful lint here.
# ruff: noqa: E501

INDEX_HTML = """\
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MapFly Live Evaluation</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #09111f;
      --panel: #111c2f;
      --line: #263650;
      --muted: #91a2bb;
      --text: #f5f8ff;
      --accent: #60a5fa;
      --ok: #34d399;
      --warn: #fbbf24;
      --bad: #fb7185;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at 10% 0%, #172b4d 0, transparent 34rem),
        var(--bg);
      color: var(--text);
      font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
    }
    main { width: min(1500px, 100%); margin: 0 auto; padding: 22px; }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 16px;
    }
    h1 { margin: 0; font-size: clamp(21px, 3vw, 30px); letter-spacing: .02em; }
    .subtitle { margin-top: 4px; color: var(--muted); }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 7px 12px;
      background: rgba(17, 28, 47, .85);
      white-space: nowrap;
    }
    .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--warn); }
    .badge.online .dot { background: var(--ok); box-shadow: 0 0 12px var(--ok); }
    .badge.offline .dot { background: var(--bad); }
    .meta {
      display: grid;
      grid-template-columns: repeat(5, minmax(130px, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }
    .tile, .panel, .details {
      border: 1px solid var(--line);
      background: rgba(17, 28, 47, .92);
      box-shadow: 0 16px 40px rgba(0, 0, 0, .16);
    }
    .tile { border-radius: 12px; padding: 11px 13px; min-width: 0; }
    .label { color: var(--muted); font-size: 12px; }
    .value {
      margin-top: 2px;
      overflow: hidden;
      font-size: 16px;
      font-weight: 650;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .views { display: grid; grid-template-columns: 2fr 1fr; gap: 14px; }
    .panel { overflow: hidden; border-radius: 14px; }
    .panel-title {
      display: flex;
      justify-content: space-between;
      padding: 10px 13px;
      border-bottom: 1px solid var(--line);
      color: var(--muted);
    }
    .canvas {
      position: relative;
      display: grid;
      min-height: 320px;
      place-items: center;
      background: #030712;
    }
    .canvas img {
      display: none;
      width: 100%;
      height: min(68vh, 720px);
      object-fit: contain;
    }
    .canvas img.ready { display: block; }
    .placeholder { color: #66758e; }
    .details {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 1px;
      margin-top: 14px;
      overflow: hidden;
      border-radius: 14px;
      background: var(--line);
    }
    .detail { min-height: 72px; padding: 12px 14px; background: var(--panel); }
    .warnings {
      display: flex;
      min-height: 30px;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }
    .warning {
      border: 1px solid color-mix(in srgb, var(--bad), transparent 35%);
      border-radius: 999px;
      padding: 4px 10px;
      background: color-mix(in srgb, var(--bad), transparent 88%);
      color: #fecdd3;
    }
    .healthy { color: var(--ok); }
    @media (max-width: 900px) {
      header { align-items: flex-start; flex-direction: column; }
      .meta { grid-template-columns: repeat(2, 1fr); }
      .views { grid-template-columns: 1fr; }
      .details { grid-template-columns: repeat(2, 1fr); }
      .canvas img { height: auto; max-height: 64vh; }
    }
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>MapFly Live Evaluation</h1>
      <div class="subtitle" id="subtitle">Latest closed-loop observations saved by the runner, not a separate UE video stream</div>
    </div>
    <div id="connection" class="badge"><span class="dot"></span><span>Connecting</span></div>
  </header>

  <section class="meta">
    <div class="tile"><div class="label">RUN</div><div id="run" class="value">—</div></div>
    <div class="tile"><div class="label">STATE</div><div id="state" class="value">Waiting</div></div>
    <div class="tile">
      <div class="label">PROGRESS</div><div id="progress" class="value">0 / 0</div>
    </div>
    <div class="tile"><div class="label">EPISODE</div><div id="episode" class="value">—</div></div>
    <div class="tile">
      <div class="label">OBSERVATION / AGE</div><div id="observation" class="value">—</div>
    </div>
  </section>

  <section class="views">
    <article class="panel">
      <div class="panel-title"><span id="fpv-title">First-person view (FPV)</span><span>Latest observation</span></div>
      <div class="canvas">
        <span id="fpv-empty" class="placeholder">Waiting for FPV frame</span><img id="fpv">
      </div>
    </article>
    <article class="panel">
      <div class="panel-title"><span>Live map</span><span>Latest observation</span></div>
      <div class="canvas">
        <span id="map-empty" class="placeholder">Waiting for map frame</span><img id="map">
      </div>
    </article>
  </section>

  <section class="details">
    <div class="detail"><div class="label">Mean brightness</div><div id="mean" class="value">—</div></div>
    <div class="detail"><div class="label">Brightness std</div><div id="std" class="value">—</div></div>
    <div class="detail">
      <div class="label">White / black pixels</div><div id="extremes" class="value">—</div>
    </div>
    <div class="detail">
      <div class="label">Last result</div><div id="result" class="value">—</div>
    </div>
  </section>
  <section id="warnings" class="warnings"><span class="healthy">No image warnings</span></section>
</main>
<script>
  const labels = {
    waiting: "Waiting for evaluation",
    running: "Running",
    complete: "Complete",
    constant_frame: "Near-constant frame",
    overexposed: "Possibly overexposed",
    underexposed: "Possibly underexposed",
    frame_stale: "No new frame for a while"
  };
  const byId = id => document.getElementById(id);
  let lastFrameKey = "";

  function text(id, value) { byId(id).textContent = value; }
  function percent(value) { return `${(100 * Number(value)).toFixed(2)}%`; }

  function showImage(id, emptyId, url, frameKey) {
    const image = byId(id);
    if (!url) {
      image.classList.remove("ready");
      byId(emptyId).style.display = "";
      return;
    }
    if (lastFrameKey !== frameKey) image.src = `${url}?v=${encodeURIComponent(frameKey)}`;
    image.classList.add("ready");
    byId(emptyId).style.display = "none";
  }

  function render(status) {
    const connection = byId("connection");
    connection.className = "badge online";
    connection.lastElementChild.textContent = "Monitor connected";
    text("run", status.run_id || "—");
    text("state", labels[status.state] || status.state);
    text("progress", `${status.completed_episodes} / ${status.total_episodes}`);
    const frame = status.frame;
    const frameKey = frame ? `${frame.episode_id}:${frame.observation_index}` : "";
    text("episode", frame?.episode_id || "—");
    text(
      "observation",
      frame ? `${frame.observation_index} / ${frame.age_s.toFixed(1)}s` : "—"
    );
    showImage("fpv", "fpv-empty", frame?.fpv_url, frameKey);
    showImage("map", "map-empty", frame?.map_url, frameKey);
    lastFrameKey = frameKey;

    const health = frame?.health;
    text("mean", health ? health.brightness_mean.toFixed(1) : "—");
    text("std", health ? health.brightness_std.toFixed(1) : "—");
    text(
      "extremes",
      health ? `${percent(health.white_fraction)} / ${percent(health.black_fraction)}` : "—"
    );

    const result = status.last_result;
    text(
      "result",
      result
        ? `${result.episode_id} · ${result.done_reason} · SR=${Number(result.success)} · `
          + `OSR=${Number(result.oracle_success)} · `
          + `NE=${Number(result.navigation_error_m).toFixed(1)}m`
        : "—"
    );

    const warnings = byId("warnings");
    warnings.replaceChildren();
    if (!status.warnings.length) {
      const healthy = document.createElement("span");
      healthy.className = "healthy";
      healthy.textContent = "No image warnings";
      warnings.appendChild(healthy);
    } else {
      status.warnings.forEach(code => {
        const warning = document.createElement("span");
        warning.className = "warning";
        warning.textContent = labels[code] || code;
        warnings.appendChild(warning);
      });
    }
  }

  async function refresh() {
    try {
      const response = await fetch("/api/status", {cache: "no-store"});
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      render(await response.json());
    } catch (error) {
      const connection = byId("connection");
      connection.className = "badge offline";
      connection.lastElementChild.textContent = "Disconnected";
    }
  }
  refresh();
  setInterval(refresh, 500);
</script>
</body>
</html>
"""

# Recording overlay at /?hud=1 and /hud: FPV left, map right, both at the policy's 224x224
# input size although closed-loop FPV is captured at 448.
HUD_HTML = """\
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MapFly HUD</title>
  <style>
    html, body {
      margin: 0;
      width: max-content;
      height: max-content;
      background: #000;
      color: #f8fafc;
      font: 700 14px/1.2 system-ui, -apple-system, "Segoe UI", sans-serif;
    }
    .hud {
      display: flex;
      flex-direction: row;
      align-items: flex-start;
      gap: 20px;
      padding: 10px;
    }
    .pane {
      display: flex;
      flex-direction: column;
      width: 224px;
      overflow: hidden;
      background: #0b0f14;
      border: 1px solid rgba(248, 250, 252, .16);
      border-radius: 10px;
      box-shadow: 0 10px 28px rgba(0, 0, 0, .55);
    }
    .label {
      box-sizing: border-box;
      width: 224px;
      padding: 8px 10px 7px;
      letter-spacing: .22em;
      text-align: center;
      color: #fff;
      background: #111827;
      border-bottom: 2px solid #fbbf24;
    }
    .frame {
      width: 224px;
      height: 224px;
      overflow: hidden;
      background: #030712;
    }
    .frame img {
      display: none;
      width: 224px;
      height: 224px;
      object-fit: fill;
    }
    .frame img.ready { display: block; }
    .placeholder {
      display: grid;
      width: 224px;
      height: 224px;
      place-items: center;
      color: #94a3b8;
      font-weight: 600;
      font-size: 14px;
    }
  </style>
</head>
<body data-layout="fpv-map">
  <div class="hud">
    <div class="pane">
      <div class="label">FPV</div>
      <div class="frame fpv">
        <span id="fpv-empty" class="placeholder">Waiting</span>
        <img id="fpv" alt="FPV">
      </div>
    </div>
    <div class="pane">
      <div class="label">MAP</div>
      <div class="frame map">
        <span id="map-empty" class="placeholder">Waiting</span>
        <img id="map" alt="Map">
      </div>
    </div>
  </div>
  <script>
    const byId = id => document.getElementById(id);
    let lastFrameKey = "";

    function showImage(id, emptyId, url, frameKey) {
      const image = byId(id);
      if (!url) {
        image.classList.remove("ready");
        byId(emptyId).style.display = "";
        return;
      }
      if (lastFrameKey !== frameKey) image.src = `${url}?v=${encodeURIComponent(frameKey)}`;
      image.classList.add("ready");
      byId(emptyId).style.display = "none";
    }

    async function refresh() {
      try {
        const response = await fetch("/api/status", {cache: "no-store"});
        if (!response.ok) return;
        const status = await response.json();
        const frame = status.frame;
        const frameKey = frame ? `${frame.episode_id}:${frame.observation_index}` : "";
        showImage("fpv", "fpv-empty", frame?.fpv_url, frameKey);
        showImage("map", "map-empty", frame?.map_url, frameKey);
        lastFrameKey = frameKey;
      } catch (error) {}
    }
    refresh();
    setInterval(refresh, 500);
  </script>
</body>
</html>
"""
