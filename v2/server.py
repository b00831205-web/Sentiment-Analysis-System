"""Flask web server for v2.

This server exposes:
- a lightweight HTML UI for interactive prediction, and
- JSON API endpoints for health checks and sentiment prediction.

The server loads previously trained artifacts from v0/v1 (no retraining).
"""

from __future__ import annotations

import os
import time
import logging
from pathlib import Path
from typing import List, Dict, Any
import sys

import math
import re
from flask import Flask, request, jsonify, render_template_string

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v2.logging_config import setup_logging
from v2.model_loader import load_v0_model, load_v1_model
from v2.predict import predict

from v0.data import ensure_aclImdb

log = logging.getLogger("v2.server")

HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>V2 Sentiment System</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root{
      --ink: #0b0f18;
      --muted: #5a6472;
      --line: rgba(11,15,24,0.12);
      --card: rgba(255,255,255,0.92);
      --shadow: 0 18px 45px rgba(11,15,24,0.08);
      --radius: 18px;
      --radius2: 14px;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas;
      --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial;
      --accent: #0b0f18;

      /* Theme variables (used by animated background layers) */
      --washA: rgba(255,255,255,0.00);
      --washB: rgba(255,255,255,0.00);
      --baseA: #ffffff;
      --baseB: #f6f7fb;

      /* bokeh colors */
      --bokeh1: rgba(56,189,248,0.18);
      --bokeh2: rgba(255,183,3,0.14);
      --bokeh3: rgba(99,102,241,0.10);
    }

    /* Background themes */
    body.theme-pos{
      --washA: rgba(90, 200, 250, 0.42);
      --washB: rgba(255, 214, 102, 0.30);
      --baseA: #f2fbff;
      --baseB: #fff7e0;

      --bokeh1: rgba(56,189,248,0.24);
      --bokeh2: rgba(255,183,3,0.20);
      --bokeh3: rgba(16,185,129,0.14);
    }
    body.theme-neg{
      --washA: rgba(56, 189, 248, 0.10);
      --washB: rgba(148, 163, 184, 0.08);
      --baseA: #0b1220;
      --baseB: #111827;

      --bokeh1: rgba(56,189,248,0.18);
      --bokeh2: rgba(148,163,184,0.14);
      --bokeh3: rgba(99,102,241,0.12);
    }

    *{ box-sizing:border-box; }
    html, body{ height:100%; }

    /* Animated "Weather-app-like" background:
       - body::before: moving soft gradient field
       - body::after: drifting bokeh blobs
       Both are fixed so no seams on scroll.
    */
    body{
      margin:0;
      font-family: var(--sans);
      color: var(--ink);
      background: transparent;
      transition: color 260ms ease;
      position: relative;
      overflow-x: hidden;
    }

    body::before{
      content:"";
      position: fixed;
      inset: -10%;
      z-index: -2;
      background:
        radial-gradient(1200px 700px at 12% 0%, var(--washA), rgba(255,255,255,0) 56%),
        radial-gradient(900px 520px at 85% 10%, var(--washB), rgba(255,255,255,0) 62%),
        radial-gradient(900px 700px at 40% 80%, rgba(99,102,241,0.10), rgba(255,255,255,0) 58%),
        linear-gradient(180deg, var(--baseA) 0%, var(--baseB) 100%);
      background-repeat: no-repeat;
      background-size: 160% 160%;
      filter: saturate(1.03) contrast(1.02);
      will-change: transform, background-position;
      animation: drift 18s ease-in-out infinite;
    }

    body::after{
      content:"";
      position: fixed;
      inset: -15%;
      z-index: -1;
      pointer-events: none;
      background:
        radial-gradient(140px 140px at 20% 25%, var(--bokeh1), rgba(255,255,255,0) 70%),
        radial-gradient(220px 220px at 70% 18%, var(--bokeh2), rgba(255,255,255,0) 72%),
        radial-gradient(180px 180px at 55% 65%, var(--bokeh3), rgba(255,255,255,0) 74%),
        radial-gradient(260px 260px at 15% 78%, rgba(255,255,255,0.10), rgba(255,255,255,0) 72%),
        radial-gradient(200px 200px at 88% 78%, rgba(255,255,255,0.08), rgba(255,255,255,0) 70%);
      filter: blur(2px);
      opacity: 0.95;
      animation: floaty 14s ease-in-out infinite;
    }

    @keyframes drift{
      0%   { transform: translate3d(0px, 0px, 0px) scale(1.02); background-position: 0% 0%; }
      50%  { transform: translate3d(-18px, 10px, 0px) scale(1.05); background-position: 100% 60%; }
      100% { transform: translate3d(0px, 0px, 0px) scale(1.02); background-position: 0% 0%; }
    }
    @keyframes floaty{
      0%   { transform: translate3d(0px, 0px, 0px); opacity: 0.85; }
      50%  { transform: translate3d(22px, -10px, 0px); opacity: 0.98; }
      100% { transform: translate3d(0px, 0px, 0px); opacity: 0.85; }
    }
    @media (prefers-reduced-motion: reduce){
      body::before, body::after{ animation: none !important; }
    }

    /* Negative theme text tweaks */
    body.theme-neg{ color: #e5e7eb; }
    body.theme-neg .brand p,
    body.theme-neg .brand .kicker,
    body.theme-neg .meta,
    body.theme-neg label,
    body.theme-neg .kpi .k,
    body.theme-neg .kpi .s{
      color: rgba(229,231,235,0.72);
    }

    a{ color: inherit; text-decoration:none; }

    .container{ max-width: 1080px; margin: 0 auto; padding: 26px 22px 40px; }

    .mast{
      display:flex; align-items:flex-start; justify-content:space-between; gap:16px;
      padding: 10px 2px 22px;
      border-bottom: 1px solid var(--line);
      margin-bottom: 18px;
    }
    .brand{ display:flex; flex-direction:column; gap: 10px; }
    .brand .kicker{
      font-size: 12px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .brand h1{
      margin:0;
      font-size: 34px;
      line-height: 1.05;
      letter-spacing: -0.02em;
      font-weight: 780;
    }
    .brand p{
      margin: 0;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.45;
      max-width: 740px;
    }

    .chip{
      border: 1px solid var(--line);
      background: var(--card);
      border-radius: 999px;
      padding: 10px 14px;
      box-shadow: 0 10px 22px rgba(11,15,24,0.06);
      display:flex; align-items:center; gap: 10px;
      height: fit-content;
      white-space: nowrap;
      backdrop-filter: blur(10px);
    }
    body.theme-neg .chip{ background: rgba(17,24,39,0.72); border-color: rgba(255,255,255,0.10); }

    .dot{
      width: 10px; height: 10px; border-radius: 999px;
      background: rgba(11,15,24,0.22);
      box-shadow: 0 0 0 6px rgba(11,15,24,0.05);
      transition: background 220ms ease, box-shadow 220ms ease;
    }
    body.theme-pos .dot{
      background: rgba(255, 183, 3, 0.95);
      box-shadow: 0 0 0 6px rgba(255, 183, 3, 0.18);
    }
    body.theme-neg .dot{
      background: rgba(56, 189, 248, 0.90);
      box-shadow: 0 0 0 6px rgba(56, 189, 248, 0.16);
    }

    .chip .label{
      font-size: 12px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--muted);
      margin-right: 2px;
    }
    body.theme-neg .chip .label{ color: rgba(229,231,235,0.70); }
    .chip .value{
      font-size: 12px;
      font-family: var(--mono);
      color: var(--ink);
    }
    body.theme-neg .chip .value{ color: #e5e7eb; }

    .stack{ display:flex; flex-direction:column; gap: 28px; margin-top: 16px; }

    .card{
      border: 1px solid var(--line);
      background: var(--card);
      border-radius: 18px;
      box-shadow: var(--shadow);
      padding: 18px;
      backdrop-filter: blur(12px);
    }
    body.theme-neg .card{
      background: rgba(17,24,39,0.72);
      border-color: rgba(255,255,255,0.10);
      box-shadow: 0 22px 70px rgba(0,0,0,0.35);
    }

    .section-head{
      display:flex; justify-content:space-between; align-items:baseline; gap: 12px;
      margin-bottom: 12px;
    }
    .section-head h2{
      margin:0;
      font-size: 18px;
      letter-spacing: -0.01em;
      font-weight: 760;
    }
    .meta{
      font-family: var(--mono);
      font-size: 12px;
      color: var(--muted);
    }

    label{
      display:block;
      font-size: 12px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--muted);
      margin: 12px 0 8px;
    }
    textarea, input[type="text"], input[type="number"], select{
      width:100%;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 10px 12px;
      background: rgba(255,255,255,0.86);
      color: var(--ink);
      outline: none;
      transition: border 140ms ease, box-shadow 140ms ease;
      font-size: 14px;
    }
    textarea{ height: 120px; resize: vertical; min-height: 90px; }
    textarea:focus, input:focus, select:focus{
      border-color: rgba(11,15,24,0.28);
      box-shadow: 0 0 0 6px rgba(11,15,24,0.06);
    }
    body.theme-neg textarea, body.theme-neg input, body.theme-neg select{
      background: rgba(255,255,255,0.06);
      border-color: rgba(255,255,255,0.10);
      color: #e5e7eb;
    }
    body.theme-neg textarea:focus, body.theme-neg input:focus, body.theme-neg select:focus{
      box-shadow: 0 0 0 6px rgba(56,189,248,0.12);
      border-color: rgba(56,189,248,0.35);
    }

    .grid2{ display:grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    @media (max-width: 860px){
      .mast{ flex-direction: column; align-items: flex-start; }
      .chip{ width: fit-content; }
      .grid2{ grid-template-columns: 1fr; }
    }

    .actions{ display:flex; gap: 10px; align-items:center; margin-top: 12px; flex-wrap:wrap; }
    .btn{
      border-radius: 999px;
      border: 1px solid rgba(11,15,24,0.14);
      padding: 10px 14px;
      font-size: 13px;
      font-weight: 700;
      letter-spacing: 0.02em;
      cursor: pointer;
      transition: transform 120ms ease, box-shadow 120ms ease, background 120ms ease;
      display:inline-flex; align-items:center; gap: 8px;
      user-select:none;
      backdrop-filter: blur(10px);
    }
    .btn.primary{
      background: var(--accent);
      color: white;
      border-color: rgba(11,15,24,0.14);
      box-shadow: 0 12px 24px rgba(11,15,24,0.18);
    }
    .btn.primary:hover{ transform: translateY(-1px); box-shadow: 0 16px 30px rgba(11,15,24,0.22); }
    .btn.ghost{
      background: rgba(255,255,255,0.72);
      color: var(--ink);
    }
    .btn.ghost:hover{ transform: translateY(-1px); box-shadow: 0 14px 26px rgba(11,15,24,0.10); }
    body.theme-neg .btn.ghost{
      background: rgba(255,255,255,0.07);
      border-color: rgba(255,255,255,0.12);
      color: #e5e7eb;
    }

    .kpi-grid{ display:grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 12px; }
    @media (max-width: 860px){ .kpi-grid{ grid-template-columns: 1fr 1fr; } }
    @media (max-width: 520px){ .kpi-grid{ grid-template-columns: 1fr; } }

    .kpi{
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.72);
      border-radius: 14px;
      padding: 12px;
      min-height: 76px;
      backdrop-filter: blur(10px);
    }
    body.theme-neg .kpi{
      background: rgba(255,255,255,0.05);
      border-color: rgba(255,255,255,0.10);
    }
    .kpi .k{
      font-size: 11px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .kpi .v{
      margin-top: 6px;
      font-size: 18px;
      font-weight: 780;
      letter-spacing: -0.01em;
    }
    .kpi .s{
      margin-top: 6px;
      font-size: 12px;
      color: var(--muted);
      font-family: var(--mono);
    }

    .badge{
      display:inline-flex;
      align-items:center;
      gap: 8px;
      border-radius: 999px;
      padding: 6px 10px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.78);
      font-size: 12px;
      font-weight: 750;
      color: inherit;
      backdrop-filter: blur(10px);
    }
    body.theme-neg .badge{
      background: rgba(255,255,255,0.06);
      border-color: rgba(255,255,255,0.10);
      color: #e5e7eb;
    }

    .sig{
      width: 10px; height: 10px; border-radius: 999px;
      box-shadow: 0 0 0 6px rgba(0,0,0,0.05);
      background: rgba(156, 163, 175, 0.9);
    }
    .sig.pos{
      background: rgba(34, 197, 94, 0.95);
      box-shadow: 0 0 0 6px rgba(34, 197, 94, 0.16);
    }
    .sig.neg{
      background: rgba(239, 68, 68, 0.95);
      box-shadow: 0 0 0 6px rgba(239, 68, 68, 0.14);
    }

    .bar{
      margin-top: 10px;
      height: 10px;
      border-radius: 999px;
      background: rgba(11,15,24,0.10);
      overflow:hidden;
      position: relative;
    }
    body.theme-neg .bar{ background: rgba(255,255,255,0.10); }
    .bar > i{
      display:block;
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, rgba(11,15,24,0.88), rgba(11,15,24,0.55));
      border-radius: 999px;
      transition: width 260ms ease;
    }
    body.theme-pos .bar > i{
      background: linear-gradient(90deg, rgba(255, 183, 3, 0.95), rgba(56, 189, 248, 0.70));
    }
    body.theme-neg .bar > i{
      background: linear-gradient(90deg, rgba(56, 189, 248, 0.85), rgba(148, 163, 184, 0.55));
    }

    .divider{ height: 1px; background: var(--line); margin: 14px 0; }

    .pillrow{ display:flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
    .pill{
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.70);
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 12px;
      font-family: var(--mono);
      color: var(--muted);
      backdrop-filter: blur(10px);
    }
    body.theme-neg .pill{
      background: rgba(255,255,255,0.06);
      border-color: rgba(255,255,255,0.10);
      color: rgba(229,231,235,0.75);
    }

    .examples{ display:grid; grid-template-columns: 1fr; gap: 12px; margin-top: 12px; }
    .ex{
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.70);
      border-radius: 14px;
      padding: 12px;
      backdrop-filter: blur(10px);
    }
    body.theme-neg .ex{
      background: rgba(255,255,255,0.05);
      border-color: rgba(255,255,255,0.10);
    }
    .ex-head{ display:flex; align-items:center; justify-content:space-between; gap: 10px; }
    .ex-head .left{ display:flex; align-items:center; gap: 10px; }
    .ex-head .p{ font-family: var(--mono); font-size: 12px; color: var(--muted); }
    body.theme-neg .ex-head .p{ color: rgba(229,231,235,0.65); }
    .ex .txt{ margin-top: 8px; font-size: 14px; line-height: 1.45; }

    .error{
      border: 1px solid rgba(239,68,68,0.22);
      background: rgba(239,68,68,0.10);
      border-radius: 14px;
      padding: 12px;
      color: inherit;
      backdrop-filter: blur(10px);
    }

    details{ margin-top: 10px; }
    pre{
      margin: 10px 0 0;
      padding: 12px;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: rgba(11,15,24,0.04);
      overflow-x: auto;
      font-size: 12px;
      font-family: var(--mono);
      color: inherit;
      backdrop-filter: blur(10px);
    }
    body.theme-neg pre{
      background: rgba(255,255,255,0.06);
      border-color: rgba(255,255,255,0.10);
    }

    .loading{
      color: var(--muted);
      font-size: 13px;
      font-family: var(--mono);
    }
  </style>
</head>

<body class="theme-neutral">
  <div class="container">
    <div class="mast">
      <div class="brand">
        <!-- changed as requested -->
        <div class="kicker">GROUP-8 PROJECT</div>
        <h1>Sentiment System</h1>
        <p>
          Consistent pipeline with v0/v1 using the Stanford IMDb dataset (auto-download + cache).
          Feature B performs keyword-based aggregation over local reviews as a stand-in for movie-level retrieval.
        </p>
      </div>
      <div class="chip">
        <span class="dot"></span>
        <span class="label">Theme</span>
        <span id="themeName" class="value">Neutral</span>
      </div>
    </div>

    <div class="stack">
      <!-- Feature A -->
      <div class="card">
        <div class="section-head">
          <h2>Feature A · Single Text Prediction</h2>
          <div id="aMeta" class="meta"></div>
        </div>

        <div class="grid2">
          <div>
            <label>Model</label>
            <select id="modelA">
              <option value="v0">v0 (LogReg + TF-IDF)</option>
              <option value="v1">v1 (2-layer NN + TF-IDF+SVD)</option>
            </select>
          </div>
          <div>
            <label>Quick examples</label>
            <select id="quickA" onchange="setExampleA()">
              <option value="" selected>Select…</option>
              <option value="pos">Positive example</option>
              <option value="neg">Negative example</option>
              <option value="mix">Mixed/neutral</option>
            </select>
          </div>
        </div>

        <label>Text</label>
        <textarea id="textA">This movie is amazing and I loved it.</textarea>

        <div class="actions">
          <button class="btn primary" onclick="runA()">Run Prediction</button>
          <button class="btn ghost" onclick="resetTheme()">Reset Theme</button>
        </div>

        <div id="outA" style="margin-top: 12px;"></div>

        <details>
          <summary class="meta">Show raw JSON</summary>
          <pre id="outA_raw">{}</pre>
        </details>
      </div>

      <!-- Feature B -->
      <div class="card">
        <div class="section-head">
          <h2>Feature B · Keyword Aggregation (Local Reviews)</h2>
          <div id="bMeta" class="meta"></div>
        </div>

        <div class="grid2">
          <div>
            <label>Model</label>
            <select id="modelB">
              <option value="v0">v0</option>
              <option value="v1">v1</option>
            </select>
          </div>
          <div>
            <label>Split</label>
            <select id="split">
              <option value="train">train</option>
              <option value="test">test</option>
              <option value="both" selected>both</option>
            </select>
          </div>
        </div>

        <div class="grid2">
          <div>
            <label>Keyword</label>
            <input id="kw" type="text" value="love" />
          </div>
          <div>
            <label>Max reviews</label>
            <input id="limit" type="number" value="60" min="10" max="300" />
          </div>
        </div>

        <div class="actions">
          <button class="btn primary" onclick="runB()">Run Analysis</button>
          <button class="btn ghost" onclick="setKeyword('great')">Try “great”</button>
          <button class="btn ghost" onclick="setKeyword('boring')">Try “boring”</button>
          <button class="btn ghost" onclick="setKeyword('worst')">Try “worst”</button>
        </div>

        <div id="outB" style="margin-top: 12px;"></div>

        <details>
          <summary class="meta">Show raw JSON</summary>
          <pre id="outB_raw">{}</pre>
        </details>
      </div>

      <!-- Feature C -->
      <div class="card">
        <div class="section-head">
          <h2>Feature C · Movie Title Recommendation (Test Reviews)</h2>
          <div id="cMeta" class="meta"></div>
        </div>

        <div class="grid2">
          <div>
            <label>Model</label>
            <select id="modelC">
              <option value="v0">v0</option>
              <option value="v1">v1</option>
            </select>
          </div>
          <div>
            <label>Max reviews (test split)</label>
            <input id="limitC" type="number" value="80" min="10" max="300" />
          </div>
        </div>

        <div style="margin-top: 10px;">
          <label>Movie title</label>
          <input id="movie" type="text" value="The Godfather" />
        </div>

        <div class="actions">
          <button class="btn primary" onclick="runC()">Run Recommendation</button>
          <button class="btn ghost" onclick="setMovie('Titanic')">Try “Titanic”</button>
          <button class="btn ghost" onclick="setMovie('Spider-Man')">Try “Spider-Man”</button>
          <button class="btn ghost" onclick="setMovie('Battlefield Earth')">Try “Battlefield Earth”</button>
          <button class="btn ghost" onclick="setMovie('Plan 9 from Outer Space')">Try “Plan 9 from Outer Space”</button>
        </div>

        <div id="outC" style="margin-top: 12px;"></div>

        <details>
          <summary class="meta">Show raw JSON</summary>
          <pre id="outC_raw">{}</pre>
        </details>
      </div>
    </div>
  </div>

<script>
function esc(s) {
  if (s === null || s === undefined) return "";
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function pct01(x) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) return "—";
  return (Number(x) * 100).toFixed(1) + "%";
}

function setThemeByLabel(label) {
  document.body.classList.remove("theme-pos", "theme-neg");
  const t = document.getElementById("themeName");
  if (label === 1) {
    document.body.classList.add("theme-pos");
    t.textContent = "Positive";
  } else if (label === 0) {
    document.body.classList.add("theme-neg");
    t.textContent = "Negative";
  } else {
    t.textContent = "Neutral";
  }
}

function resetTheme() { setThemeByLabel(null); }

function badge(label) {
  if (label === 1) return '<span class="badge"><span class="sig pos"></span> Positive</span>';
  if (label === 0) return '<span class="badge"><span class="sig neg"></span> Negative</span>';
  return '<span class="badge"><span class="sig"></span> Unknown</span>';
}

function bar(widthPct) {
  const w = Math.max(0, Math.min(100, Math.round(widthPct)));
  return `<div class="bar"><i style="width:${w}%"></i></div>`;
}

function setExampleA(){
  const sel = document.getElementById("quickA");
  const el = document.getElementById("textA");
  const which = sel.value;
  if (!which) return;

  if (which === "pos") el.value = "This movie is amazing and I loved it. Great acting, great story.";
  if (which === "neg") el.value = "Terrible movie. Boring plot and a complete waste of time.";
  if (which === "mix") el.value = "Not bad overall. Some scenes are good but the pacing is slow.";
}

function setKeyword(k){
  document.getElementById("kw").value = k;
}

function renderA(j){
  if (j.error) return `<div class="error">${esc(j.error)}</div>`;

  const label = j.label;
  const prob = (j.prob_pos === null || j.prob_pos === undefined) ? null : Number(j.prob_pos);
  const conf = (prob === null) ? "—" : pct01(prob);
  const model = esc(j.model || "");
  const modelName = esc(j.model_name || "");
  const latency = (j.latency_ms !== undefined) ? `${esc(j.latency_ms)} ms` : "—";

  const headline = (label === 1) ? "Positive sentiment" : "Negative sentiment";
  const sub = (label === 1)
    ? "Model indicates the text expresses positive sentiment."
    : "Model indicates the text expresses negative sentiment.";

  const barPct = (prob === null) ? 0 : prob * 100;

  return `
    <div class="kpi-grid">
      <div class="kpi">
        <div class="k">Result</div>
        <div class="v">${badge(label)} <span style="margin-left:8px;">${headline}</span></div>
        <div class="s">${esc(sub)}</div>
      </div>
      <div class="kpi">
        <div class="k">Confidence</div>
        <div class="v">${conf}</div>
        ${bar(barPct)}
        <div class="s">P(positive)</div>
      </div>
      <div class="kpi">
        <div class="k">Model</div>
        <div class="v"><span style="font-family: var(--mono); font-size: 14px;">${model}</span></div>
        <div class="s">${modelName}</div>
      </div>
      <div class="kpi">
        <div class="k">Latency</div>
        <div class="v">${latency}</div>
        <div class="s">end-to-end inference</div>
      </div>
    </div>
  `;
}


function exCards(arr, title){
  if (!arr || !arr.length) return `<div class="meta">No examples.</div>`;
  return `
    <div class="examples">
      ${arr.map(ex => {
        const lbl = ex.label;
        const p = Number(ex.prob_pos);
        const probTxt = Number.isNaN(p) ? "—" : pct01(p);
        const snip = esc(ex.snippet || "");
        return `
          <div class="ex">
            <div class="ex-head">
              <div class="left">${badge(lbl)} <span class="p">P(pos) ${probTxt}</span></div>
              <div class="p">${esc(title)}</div>
            </div>
            <div class="txt">${snip}</div>
          </div>
        `;
      }).join("")}
    </div>
  `;
}

function renderB(j){
  if (j.error) return `<div class="error">${esc(j.error)}</div>`;
  if (j.message && j.n_reviews === 0) return `<div class="error">${esc(j.message)}</div>`;

  const kw = esc(j.keyword || "");
  const split = esc(j.split || "");
  const model = esc(j.model || "");
  const n = esc(j.n_reviews || 0);

  const avgProb = (j.avg_prob_pos === null || j.avg_prob_pos === undefined) ? null : Number(j.avg_prob_pos);
  const posRatio = (j.pos_ratio === null || j.pos_ratio === undefined) ? null : Number(j.pos_ratio);

  const avgProbTxt = (avgProb === null) ? "—" : pct01(avgProb);
  const posRatioTxt = (posRatio === null) ? "—" : pct01(posRatio);
  const latency = (j.latency_ms !== undefined) ? `${esc(j.latency_ms)} ms` : "—";
  const note = esc(j.note || "");

  const keys = Array.isArray(j.top_keywords) ? j.top_keywords : [];
  const pills = keys.slice(0, 18).map(k => `<span class="pill">${esc(k)}</span>`).join("");

  const best = Array.isArray(j.best_examples) ? j.best_examples : [];
  const worst = Array.isArray(j.worst_examples) ? j.worst_examples : [];

  const posBar = (posRatio === null) ? 0 : posRatio * 100;

  return `
    <div class="kpi-grid">
      <div class="kpi">
        <div class="k">Query</div>
        <div class="v"><span style="font-family: var(--mono); font-size: 14px;">${kw}</span></div>
        <div class="s">split=${split} • model=${model}</div>
      </div>
      <div class="kpi">
        <div class="k">Reviews</div>
        <div class="v">${n}</div>
        <div class="s">matched in local dataset</div>
      </div>
      <div class="kpi">
        <div class="k">Positive ratio</div>
        <div class="v">${posRatioTxt}</div>
        ${bar(posBar)}
        <div class="s">share of label=1 • latency=${latency}</div>
      </div>
    </div>

    <div class="divider"></div>

    <div class="meta" style="margin-bottom: 6px;">
      ${note ? ("Note: " + note) : ""}
    </div>

    <div class="divider"></div>

    <div class="meta">Best examples (most positive)</div>
    ${exCards(best, "best")}

    <div class="meta" style="margin-top: 14px;">Worst examples (most negative)</div>
    ${exCards(worst, "worst")}
  `;
}

function renderC(j){
  if (j.error) return `<div class="error">${esc(j.error)}</div>`;
  if (j.message && j.n_reviews === 0) return `<div class="error">${esc(j.message)}</div>`;

  const movie = esc(j.movie || "");
  const model = esc(j.model || "");
  const n = esc(j.n_reviews || 0);

  const avgProb = (j.avg_prob_pos === null || j.avg_prob_pos === undefined) ? null : Number(j.avg_prob_pos);
  const posRatio = (j.pos_ratio === null || j.pos_ratio === undefined) ? null : Number(j.pos_ratio);
  const recIdx = (j.recommend_index_0_100 === null || j.recommend_index_0_100 === undefined) ? null : Number(j.recommend_index_0_100);

  const avgProbTxt = (avgProb === null) ? "—" : pct01(avgProb);
  const posRatioTxt = (posRatio === null) ? "—" : pct01(posRatio);
  const recTxt = (recIdx === null) ? "—" : `${recIdx} / 100`;
  const latency = (j.latency_ms !== undefined) ? `${esc(j.latency_ms)} ms` : "—";
  const note = esc(j.note || "");
  const warning = esc(j.warning || "");

  const vars = Array.isArray(j.match_variants) ? j.match_variants : [];
  const vpills = vars.slice(0, 18).map(k => `<span class="pill">${esc(k)}</span>`).join("");

  const best = Array.isArray(j.best_examples) ? j.best_examples : [];
  const worst = Array.isArray(j.worst_examples) ? j.worst_examples : [];

  const avgBar = (avgProb === null) ? 0 : avgProb * 100;
  const posBar = (posRatio === null) ? 0 : posRatio * 100;
  const recBar = (recIdx === null) ? 0 : recIdx;

  const keys = Array.isArray(j.top_keywords) ? j.top_keywords : [];
  const pills = keys.slice(0, 18).map(k => `<span class="pill">${esc(k)}</span>`).join("");

  return `
    <div class="hint">
      <b>Movie:</b> <span class="mono">${movie}</span>
      <span class="pill">test split</span>
      <span class="pill">${esc(n)} reviews</span>
      <span class="pill">model=${model}</span>
      ${warning ? `<div style="margin-top:6px;"><span class="pill">⚠ ${warning}</span></div>` : ""}
      ${note ? `<div style="margin-top:6px;" class="meta">${note}</div>` : ""}
    </div>

    <div class="kpi-grid">
      <div class="kpi">
        <div class="k">Avg P(positive)</div>
        <div class="v">${avgProbTxt}</div>
        ${bar(avgBar)}
        <div class="s">mean predicted probability</div>
      </div>
      <div class="kpi">
        <div class="k">Positive ratio</div>
        <div class="v">${posRatioTxt}</div>
        ${bar(posBar)}
        <div class="s">share of label=1</div>
      </div>
      <div class="kpi">
        <div class="k">Recommendation</div>
        <div class="v">${recTxt}</div>
        ${bar(recBar)}
        <div class="s">latency=${latency}</div>
      </div>
    </div>

    <div class="divider"></div>

    <div class="meta" style="margin-top: 10px;">Top words</div>
    <div class="pillrow">${pills || '<span class="meta">No words extracted.</span>'}</div>

    <div class="divider"></div>

    <div class="meta">Best examples (most positive)</div>
    ${exCards(best, "best")}

    <div class="meta" style="margin-top: 14px;">Worst examples (most negative)</div>
    ${exCards(worst, "worst")}
  `;
}

async function runA(){
  const model = document.getElementById("modelA").value;
  const text = document.getElementById("textA").value;

  const out = document.getElementById("outA");
  const raw = document.getElementById("outA_raw");
  const meta = document.getElementById("aMeta");

  out.innerHTML = `<div class="loading">Running prediction…</div>`;
  meta.textContent = "";

  const r = await fetch("/predict", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({model, text})
  });
  const j = await r.json();
  raw.textContent = JSON.stringify(j, null, 2);

  if (!j.error) {
    const lp = j.model_path ? String(j.model_path).split("/").slice(-1)[0] : "";
    meta.textContent = (lp ? ("artifact=" + lp + " • ") : "") + ("latency=" + (j.latency_ms ?? "—") + "ms");
    setThemeByLabel(j.label);
  }
  out.innerHTML = renderA(j);
}

async function runB(){
  const model = document.getElementById("modelB").value;
  const keyword = document.getElementById("kw").value;
  const split = document.getElementById("split").value;
  const limit = parseInt(document.getElementById("limit").value || "60", 10);

  const out = document.getElementById("outB");
  const raw = document.getElementById("outB_raw");
  const meta = document.getElementById("bMeta");

  out.innerHTML = `<div class="loading">Running analysis…</div>`;
  meta.textContent = "";

  const r = await fetch("/analyze", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({model, keyword, split, limit})
  });
  const j = await r.json();
  raw.textContent = JSON.stringify(j, null, 2);

  if (!j.error) meta.textContent = "latency=" + (j.latency_ms ?? "—") + "ms";
  out.innerHTML = renderB(j);
}

function setMovie(x){
  document.getElementById("movie").value = x;
}

async function runC(){
  const model = document.getElementById("modelC").value;
  const movie = document.getElementById("movie").value;
  const limit = parseInt(document.getElementById("limitC").value || "80", 10);

  const out = document.getElementById("outC");
  const raw = document.getElementById("outC_raw");
  const meta = document.getElementById("cMeta");

  out.innerHTML = `<div class="loading">Running recommendation…</div>`;
  meta.textContent = "";
  raw.textContent = "";

  try{
    const r = await fetch("/analyze_movie", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({model, movie, limit})
    });

    const txt = await r.text();
    raw.textContent = txt;

    let j = null;
    try{
      j = JSON.parse(txt);
    }catch(e){
      out.innerHTML = `<div class="error">Failed to parse JSON response: ${esc(String(e))}</div>`;
      return;
    }

    if (!j.error) meta.textContent = "latency=" + (j.latency_ms ?? "—") + "ms";
    out.innerHTML = renderC(j);
  }catch(err){
    out.innerHTML = `<div class="error">${esc(String(err))}</div>`;
  }

}

setThemeByLabel(null);
</script>
</body>
</html>

"""

def _safe_float(x: Any) -> Any:
    """Convert numeric to JSON-safe float; NaN/inf -> None."""
    try:
        v = float(x)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v

def _iter_review_files(acl_dir: str, split: str) -> List[Path]:
    """List review text files for a given IMDb split.

    Args:
        acl_dir: Path to the extracted `aclImdb/` directory.
        split: Dataset split name ("train" or "test").

    Returns:
        List of `.txt` review file paths (pos + neg).
    """
    root = Path(acl_dir) / split
    files: List[Path] = []
    for sub in ["pos", "neg"]:
        d = root / sub
        if d.exists():
            files.extend([p for p in d.iterdir() if p.suffix.lower() == ".txt"])
    return files

def _match_reviews(acl_dir: str, split: str, keyword: str, limit: int) -> List[str]:
    """Retrieve raw reviews that contain a keyword substring.

    Args:
        acl_dir: Path to `aclImdb/`.
        split: "train" or "test".
        keyword: Case-insensitive substring to search for.
        limit: Maximum number of reviews to return.

    Returns:
        List of raw review texts.
    """
    keyword_l = keyword.lower().strip()
    if not keyword_l:
        return []
    out: List[str] = []
    for p in _iter_review_files(acl_dir, split):
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if keyword_l in txt.lower():
            out.append(txt)
            if len(out) >= limit:
                break
    return out

_WORDISH_MOVIE = re.compile(r"[^a-z0-9]+")

def _normalize_text_movie(s: str) -> str:
    """Normalize text for robust movie-title matching.

    Args:
        s: Input string.

    Returns:
        Normalized string.
    """

    # Lowercase and replace non-alphanumeric runs with a single space.
    s = s.lower()
    s = _WORDISH_MOVIE.sub(" ", s)
    s = " ".join(s.split())
    return s

def _movie_variants(movie: str) -> List[str]:
    """Generate normalized variants of a movie title to improve recall.

    Args:
        movie: Movie title string.

    Returns:
        List of normalized variants (may be empty).
    """

    #Generate normalized variants to increase recall (e.g., 'Spider-Man' -> 'spider man', 'spiderman').
    base = _normalize_text_movie(movie)
    if not base:
        return []
    out = {base}
    out.add(base.replace(" ", ""))
    toks = base.split()
    if len(toks) >= 2:
        out.add(" ".join(toks[:2]))
        out.add(" ".join(toks[-2:]))
    return [x for x in out if x]

def _match_movie_test_only(acl_dir: str, movie: str, limit: int) -> List[str]:
    """Retrieve test-set reviews that mention a movie title.

    Args:
        acl_dir: Path to `aclImdb/`.
        movie: Movie title query.
        limit: Maximum number of reviews to return.

    Returns:
        List of raw review texts from the test split.
    """

    #Movie-level retrieval: test split only; robust phrase matching on normalized text.
    variants = _movie_variants(movie)
    if not variants:
        return []
    out: List[str] = []
    for p in _iter_review_files(acl_dir, "test"):
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        norm = " " + _normalize_text_movie(txt) + " "
        hit = False
        for q in variants:
            if f" {q} " in norm:
                hit = True
                break
        if hit:
            out.append(txt)
            if len(out) >= limit:
                break
    return out

def _top_keywords_from_vectorizer(vec, texts: List[str], topk: int = 12) -> List[str]:
    """Extract top-weighted keywords from a vectorizer for a given set of texts.

    Args:
        vec: A fitted sklearn vectorizer with `get_feature_names_out`.
        texts: List of raw input texts to analyze.
        topk: Number of keywords to return.

    Returns:
        List of keyword strings (length <= topk).
    """

    try:
        X = vec.transform(texts)
        mean = X.mean(axis=0)
        arr = mean.A1
        if arr.size == 0:
            return []
        idx = arr.argsort()[::-1][:topk]
        feats = vec.get_feature_names_out()
        return [str(feats[i]) for i in idx]
    except Exception:
        return []



def _snip(text: str, n: int = 220) -> str:
    """Create a short preview snippet from a longer text.

    Args:
        text: Full text.
        n: Maximum characters to keep.

    Returns:
        Possibly-truncated string ending with "..." if truncated.
    """
    t = " ".join(text.strip().split())
    return t if len(t) <= n else t[:n] + "..."

def create_app() -> Flask:
    """Create and configure the Flask application.

    Returns:
        Configured Flask `app` instance ready to run.

    Side Effects:
        Configures logging and attempts to load v0/v1 artifacts and the IMDb dataset.
    """
    setup_logging()
    app = Flask(__name__)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    v0_ctx = None
    v1_ctx = None
    try:
        v0_ctx = load_v0_model(project_root)
    except Exception as e:
        log.warning(f"v0 not loaded: {e}")
    try:
        v1_ctx = load_v1_model(project_root)
    except Exception as e:
        log.warning(f"v1 not loaded: {e}")

    acl_dir = ensure_aclImdb()
    log.info(f"aclImdb dir: {acl_dir}")

    @app.get("/")
    def index():
        return render_template_string(HTML)

    @app.get("/health")
    def health():
        return jsonify({
            "status": "ok",
            "v0_loaded": v0_ctx is not None,
            "v1_loaded": v1_ctx is not None,
            "acl_dir": acl_dir,
        })

    @app.post("/predict")
    def api_predict():
        payload = request.get_json(force=True)
        model = payload.get("model", "v0")
        text = payload.get("text", "")
        if not isinstance(text, str) or not text.strip():
            return jsonify({"error": "text is required"}), 400
        t0 = time.time()
        out = predict(model, v0_ctx, v1_ctx, text)
        out["latency_ms"] = int((time.time() - t0) * 1000)
        return jsonify(out)

    @app.post("/analyze_keyword")
    @app.post("/analyze")
    def api_analyze():
        payload = request.get_json(force=True)
        model = str(payload.get("model", "v0")).strip()
        keyword = str(payload.get("keyword", "")).strip()
        split = str(payload.get("split", "both")).strip().lower()
        limit = int(payload.get("limit", 60))
        limit = max(10, min(limit, 300))

        if not keyword:
            return jsonify({"error": "keyword is required"}), 400
        if split not in ("train", "test", "both"):
            return jsonify({"error": "split must be one of: train, test, both"}), 400

        t0 = time.time()
        texts: List[str] = []
        if split in ("train", "both"):
            texts.extend(_match_reviews(acl_dir, "train", keyword, limit))
        if split in ("test", "both") and len(texts) < limit:
            texts.extend(_match_reviews(acl_dir, "test", keyword, limit - len(texts)))

        if not texts:
            return jsonify({
                "model": model,
                "keyword": keyword,
                "split": split,
                "n_reviews": 0,
                "message": "No matched reviews in local dataset. Try a more common keyword (e.g., love, boring, great, worst).",
            })

        preds: List[Dict[str, Any]] = []
        probs: List[float] = []
        for tx in texts:
            out = predict(model, v0_ctx, v1_ctx, tx)
            p = out.get("prob_pos")
            if p is None:
                p = float(out["label"])
            preds.append({"label": int(out["label"]), "prob_pos": _safe_float(p), "snippet": _snip(tx)})
            probs.append(_safe_float(p) if _safe_float(p) is not None else 0.0)

        avg_prob = _safe_float(sum(probs) / len(probs))
        pos_ratio = _safe_float(sum(1 for x in preds if x["label"] == 1) / len(preds))
        keywords: List[str] = []
        if model.lower() == "v0" and v0_ctx is not None:
            try:
                pipe = v0_ctx["model"]
                vec = pipe.named_steps.get("tfidf")
                if vec is not None:
                    keywords = _top_keywords_from_vectorizer(vec, texts, topk=12)
            except Exception:
                pass
        elif model.lower() == "v1" and v1_ctx is not None:
            try:
                vec = v1_ctx["bundle"]["vectorizer"]
                keywords = _top_keywords_from_vectorizer(vec, texts, topk=12)
            except Exception:
                pass

        preds_sorted = sorted(preds, key=lambda d: d["prob_pos"])
        worst = preds_sorted[:3]
        best = preds_sorted[-3:][::-1]

        resp = {
            "model": model,
            "keyword": keyword,
            "split": split,
            "n_reviews": len(texts),
"avg_prob_pos": avg_prob,
            "pos_ratio": pos_ratio,
            "top_keywords": keywords,
            "best_examples": best,
            "worst_examples": worst,
            "latency_ms": int((time.time() - t0) * 1000),
            "note": "Local dataset keyword match (not real movie-title lookup).",
        }
        return jsonify(resp)


    @app.post("/analyze_movie")
    def analyze_movie():
        payload = request.get_json(force=True)
        model = str(payload.get("model", "v0")).strip()
        movie = str(payload.get("movie", "")).strip()
        limit = int(payload.get("limit", 80))
        limit = max(10, min(limit, 300))

        if not movie:
            return jsonify({"error": "movie title is required"}), 400

        t0 = time.time()
        texts = _match_movie_test_only(acl_dir, movie, limit)

        if not texts:
            return jsonify({
                "model": model,
                "movie": movie,
                "split": "test",
                "n_reviews": 0,
                "message": "No matched reviews in local test dataset for this movie title. Try adding more words, removing punctuation, or using a more common title.",
            })

        preds: List[Dict[str, Any]] = []
        probs: List[float] = []
        for tx in texts:
            out = predict(model, v0_ctx, v1_ctx, tx)
            p = out.get("prob_pos")
            if p is None:
                p = float(out["label"])
            preds.append({"label": int(out["label"]), "prob_pos": _safe_float(p), "snippet": _snip(tx)})
            probs.append(_safe_float(p) if _safe_float(p) is not None else 0.0)

        avg_prob = _safe_float(sum(probs) / len(probs))
        pos_ratio = _safe_float(sum(1 for x in preds if x["label"] == 1) / len(preds))
        recommend_index = int(round((avg_prob or 0.0) * 100))

        # Top words for the matched movie reviews (mirrors keyword analysis logic)
        keywords: List[str] = []
        if model.lower() == "v0" and v0_ctx is not None:
            try:
                pipe = v0_ctx["model"]
                vec = pipe.named_steps.get("tfidf")
                if vec is not None:
                    keywords = _top_keywords_from_vectorizer(vec, texts, topk=12)
            except Exception:
                pass
        elif model.lower() == "v1" and v1_ctx is not None:
            try:
                vec = v1_ctx["bundle"]["vectorizer"]
                keywords = _top_keywords_from_vectorizer(vec, texts, topk=12)
            except Exception:
                pass

        best = sorted(preds, key=lambda x: x.get("prob_pos", 0.0), reverse=True)[:5]
        worst = sorted(preds, key=lambda x: x.get("prob_pos", 0.0))[:5]

        warn = None
        if len(movie.strip()) < 4:
            warn = "Movie title is very short; matches may be noisy. Consider adding more words or a year."

        resp = {
            "mode": "movie-title",
            "movie": movie,
            "split": "test",
            "model": model,
            "n_reviews": len(texts),
            "avg_prob_pos": avg_prob,
            "pos_ratio": pos_ratio,
            "recommend_index_0_100": recommend_index,
            "top_keywords": keywords,
            "best_examples": best,
            "worst_examples": worst,
            "note": "Movie-level recommendation uses test split only (proxy for unseen user reviews).",
            "warning": warn,
            "match_variants": _movie_variants(movie),
            "latency_ms": int((time.time() - t0) * 1000),
        }
        return jsonify(resp)

    return app
