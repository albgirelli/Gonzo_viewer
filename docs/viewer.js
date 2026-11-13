// Gonzo tracer viewer: three linked orthogonal slices of the 4D concentration data,
// plus regional enhancement curves. Volumes are gzipped raw arrays in RAS order,
// index = (x * ny + y) * nz + z.

const state = {
  meta: null, t1: null, labels: null, conc: [], regions: null, dti: null,
  t: 2, cursor: [0, 0, 0], mode: "conc", vmax: 0.25, alpha: 0.85,
};

const $ = (id) => document.getElementById(id);

async function loadGz(url, Type) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  const buf = await res.arrayBuffer();
  const bytes = new Uint8Array(buf, 0, 2);
  if (bytes[0] !== 0x1f || bytes[1] !== 0x8b) return new Type(buf);  // server already decompressed it
  const stream = new Blob([buf]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Type(await new Response(stream).arrayBuffer());
}

// Perceptually ordered colormap (inferno-like control points).
const CMAP = [
  [0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99],
  [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164],
];
function cmap(v) {
  const x = Math.min(Math.max(v, 0), 1) * (CMAP.length - 1);
  const i = Math.min(Math.floor(x), CMAP.length - 2), f = x - i;
  return CMAP[i].map((c, k) => c + f * (CMAP[i + 1][k] - c));
}
// Viridis-like control points for DTI scalar maps.
const VIRIDIS = [
  [68, 1, 84], [70, 50, 127], [54, 92, 141], [39, 127, 142],
  [31, 161, 135], [74, 194, 109], [159, 218, 58], [253, 231, 37],
];
function viridis(v) {
  const x = Math.min(Math.max(v, 0), 1) * (VIRIDIS.length - 1);
  const i = Math.min(Math.floor(x), VIRIDIS.length - 2), f = x - i;
  return VIRIDIS[i].map((c, k) => c + f * (VIRIDIS[i + 1][k] - c));
}
// Display ranges for the DTI scalar maps.
const DTI_RANGE = { md: [0, 2e-3], fa: [0, 1] };
const isDti = (mode) => mode === "md" || mode === "fa" || mode === "dir";
function dtiAt(kind, i) {
  const q = state.dti[kind][i], m = state.meta.dti[kind];
  return q ? q * m.scale + m.offset : NaN;
}

function labelColor(l) {
  // Deterministic pseudo-random colour per label.
  const h = (l * 2654435761) >>> 0;
  return [h & 255, (h >> 8) & 255, (h >> 16) & 255];
}

function concAt(t, i) {
  const c = state.meta.conc;
  return state.conc[t][i] * c.scale + c.offset;
}
function index(x, y, z) {
  const [, ny, nz] = state.meta.shape;
  return (x * ny + y) * nz + z;
}

// Each plane maps canvas (col, row) to voxel (x, y, z). Radiological convention:
// patient right on the viewer's left; anterior/superior at the top.
const PLANES = {
  axial: {
    size: ([nx, ny]) => [nx, ny],
    voxel: (c, r, [nx, ny], cur) => [nx - 1 - c, ny - 1 - r, cur[2]],
    pixel: ([x, y], [nx, ny]) => [nx - 1 - x, ny - 1 - y],
  },
  coronal: {
    size: ([nx, , nz]) => [nx, nz],
    voxel: (c, r, [nx, , nz], cur) => [nx - 1 - c, cur[1], nz - 1 - r],
    pixel: ([x, , z], [nx, , nz]) => [nx - 1 - x, nz - 1 - z],
  },
  sagittal: {
    size: ([, ny, nz]) => [ny, nz],
    voxel: (c, r, [, ny, nz], cur) => [cur[0], ny - 1 - c, nz - 1 - r],
    pixel: ([, y, z], [, ny, nz]) => [ny - 1 - y, nz - 1 - z],
  },
};

function drawPlane(name) {
  const plane = PLANES[name], shape = state.meta.shape;
  const [w, h] = plane.size(shape);
  const off = drawPlane.buffers[name] ||= document.createElement("canvas");
  off.width = w; off.height = h;
  const ctx = off.getContext("2d");
  const img = ctx.createImageData(w, h);
  const { mode, vmax, alpha, t } = state;
  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      const [x, y, z] = plane.voxel(c, r, shape, state.cursor);
      const i = index(x, y, z), p = 4 * (r * w + c);
      const g = state.t1[i];
      let rgb = [g, g, g];
      if (mode === "labels" && state.labels[i]) {
        const lc = labelColor(state.labels[i]);
        rgb = rgb.map((v, k) => 0.45 * v + 0.55 * lc[k]);
      } else if (isDti(mode) && state.dti && state.dti.fa[i]) {
        let cc;
        if (mode === "dir") {
          const fa = dtiAt("fa", i), d = state.dti.dir;
          cc = [d[3 * i], d[3 * i + 1], d[3 * i + 2]].map((v) => Math.min(255, v * fa * 1.6));
        } else {
          const [lo, hi] = DTI_RANGE[mode];
          cc = viridis((dtiAt(mode, i) - lo) / (hi - lo));
        }
        rgb = rgb.map((u, k) => (1 - alpha) * u + alpha * cc[k]);
      } else if (mode === "conc" && state.labels[i]) {
        const v = concAt(t, i) / vmax;
        if (v > 0.02) {
          const cc = cmap(v), a = alpha * Math.min(1, v * 4);
          rgb = rgb.map((u, k) => (1 - a) * u + a * cc[k]);
        }
      }
      img.data[p] = rgb[0]; img.data[p + 1] = rgb[1]; img.data[p + 2] = rgb[2]; img.data[p + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);

  const canvas = $(name), dpr = window.devicePixelRatio || 1;
  const side = canvas.clientWidth;
  canvas.width = side * dpr; canvas.height = side * dpr;
  const out = canvas.getContext("2d");
  out.fillStyle = "#000"; out.fillRect(0, 0, canvas.width, canvas.height);
  const scale = Math.min(canvas.width / w, canvas.height / h);
  const ox = (canvas.width - w * scale) / 2, oy = (canvas.height - h * scale) / 2;
  out.imageSmoothingEnabled = true;
  out.drawImage(off, ox, oy, w * scale, h * scale);
  canvas._geom = { ox, oy, scale, w, h };

  const [pc, pr] = plane.pixel(state.cursor, shape);
  out.strokeStyle = "rgba(80,200,255,0.7)"; out.lineWidth = dpr;
  out.beginPath();
  out.moveTo(ox + (pc + 0.5) * scale, oy); out.lineTo(ox + (pc + 0.5) * scale, oy + h * scale);
  out.moveTo(ox, oy + (pr + 0.5) * scale); out.lineTo(ox + w * scale, oy + (pr + 0.5) * scale);
  out.stroke();
  out.fillStyle = "#bbb"; out.font = `${12 * dpr}px system-ui`;
  if (name !== "sagittal") { out.fillText("R", 6 * dpr, canvas.height - 8 * dpr); out.fillText("L", canvas.width - 14 * dpr, canvas.height - 8 * dpr); }
  else { out.fillText("A", 6 * dpr, canvas.height - 8 * dpr); out.fillText("P", canvas.width - 14 * dpr, canvas.height - 8 * dpr); }
}
drawPlane.buffers = {};

function drawAll() {
  for (const n of Object.keys(PLANES)) drawPlane(n);
}

function onClick(name, ev) {
  const canvas = $(name), g = canvas._geom, dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const c = Math.floor(((ev.clientX - rect.left) * dpr - g.ox) / g.scale);
  const r = Math.floor(((ev.clientY - rect.top) * dpr - g.oy) / g.scale);
  if (c < 0 || r < 0 || c >= g.w || r >= g.h) return;
  state.cursor = PLANES[name].voxel(c, r, state.meta.shape, state.cursor);
  drawAll();
  updateVoxelInfo();
}

function plotTheme() {
  const cs = getComputedStyle(document.documentElement);
  const text = cs.getPropertyValue("--text").trim(), grid = cs.getPropertyValue("--border").trim();
  return {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: text, family: "system-ui, sans-serif" },
    xaxis: { title: "Hours after injection", gridcolor: grid, zerolinecolor: grid },
    yaxis: { gridcolor: grid, zerolinecolor: grid },
    margin: { l: 60, r: 10, t: 30, b: 90 }, legend: { orientation: "h", y: -0.32 },
  };
}

function regionByLabel(l) {
  return state.regions.regions.find((r) => r.label === l);
}

function updateVoxelInfo() {
  const [x, y, z] = state.cursor, i = index(x, y, z);
  const hours = state.meta.sessions.map((s) => s.hours);
  const label = state.labels[i];
  const region = label ? regionByLabel(label) : null;
  const values = state.conc.map((_, t) => concAt(t, i));
  const A = state.meta.affine;
  const mm = A.slice(0, 3).map((row) => (row[0] * x + row[1] * y + row[2] * z + row[3]).toFixed(0));
  $("voxelInfo").innerHTML = `
    <b>Voxel</b> (${x}, ${y}, ${z}) · RAS ${mm.join(", ")} mm<br>
    <b>Region</b> ${region ? region.name : label ? `label ${label}` : "outside segmentation"}<br>
    <table><tr><th>Time</th><th>mmol/L</th></tr>
    ${state.meta.sessions.map((s, t) => `<tr><td>${s.label}</td><td>${label ? values[t].toFixed(3) : "–"}</td></tr>`).join("")}
    </table>${dtiInfo(i, region)}`;
  const traces = [];
  if (label) traces.push({ x: hours, y: values, name: "this voxel", mode: "lines+markers" });
  if (region) traces.push({ x: hours, y: region.mean, name: `${region.name} (mean)`, mode: "lines+markers", line: { dash: "dot" } });
  const layout = plotTheme();
  layout.yaxis.title = "Concentration (mmol/L)";
  Plotly.react("voxelPlot", traces, layout, { responsive: true, displaylogo: false });
}

function dtiInfo(i, region) {
  if (!state.dti) return "";
  const fmt = (v, d) => (Number.isFinite(v) ? v.toFixed(d) : "–");
  const here = state.dti.fa[i]
    ? `MD ${fmt(dtiAt("md", i) * 1e3, 2)}×10⁻³ mm²/s · FA ${fmt(dtiAt("fa", i), 2)}`
    : "no DTI here (DTI covers brain tissue only)";
  const reg = region && region.md != null ? `<br>Region mean: MD ${fmt(region.md, 2)}×10⁻³ mm²/s · FA ${fmt(region.fa, 2)}` : "";
  return `<p><b>DTI</b> ${here}${reg}</p>`;
}

function updateColorbar() {
  const cb = $("cbar").getContext("2d"), mode = state.mode;
  cb.clearRect(0, 0, 180, 12);
  for (let i = 0; i < 180; i++) {
    const v = i / 179;
    const c = mode === "dir" ? [[230, 40, 40], [40, 200, 60], [60, 90, 255]][Math.min(2, Math.floor(3 * v))]
      : isDti(mode) ? viridis(v) : cmap(v);
    cb.fillStyle = `rgb(${c.join(",")})`; cb.fillRect(i, 0, 1, 12);
  }
  $("vmaxLabel").style.display = isDti(mode) ? "none" : "";
  $("timeButtons").style.opacity = isDti(mode) ? 0.4 : 1;
  $("cbar").parentElement.style.display = mode === "t1" || mode === "labels" ? "none" : "";
  const [lo, hi] = {
    md: ["0", "2.0×10⁻³ mm²/s"], fa: ["FA 0", "1"], dir: ["", "red left–right · green front–back · blue up–down"],
  }[mode] || ["0", `${state.vmax.toFixed(2)} mmol/L`];
  $("cbarMin").textContent = lo; $("cbarMax").textContent = hi;
}

async function loadDti() {
  if (state.dti) return state.dti;
  loadDti.pending ||= (async () => {
    const d = state.meta.dti;
    const [md, fa, dir] = await Promise.all([d.md, d.fa, d.dir].map((x) => loadGz("data/" + x.file, Uint8Array)));
    state.dti = { md, fa, dir };
    updateVoxelInfo();
    return state.dti;
  })();
  return loadDti.pending;
}

function updateRegionPlot() {
  const q = $("quantity").value;
  const chosen = [...$("regionSelect").selectedOptions].map((o) => +o.value);
  const hours = state.meta.sessions.map((s) => s.hours);
  const traces = chosen.map((l) => {
    const r = regionByLabel(l);
    return { x: hours, y: q === "mean" ? r.mean : r.amount_umol, name: r.name, mode: "lines+markers" };
  });
  const layout = plotTheme();
  layout.yaxis.title = q === "mean" ? "Concentration (mmol/L)" : "Tracer amount (µmol)";
  if ($("showBlood").checked) {
    const b = state.regions.blood;
    traces.push({ x: b.hours, y: b.conc, name: "blood plasma", mode: "lines+markers", yaxis: "y2", line: { color: "#c0392b", dash: "dash" } });
    layout.yaxis2 = { title: "Plasma (mmol/L)", overlaying: "y", side: "right", showgrid: false };
    layout.margin.r = 60;
  }
  Plotly.react("regionPlot", traces, layout, { responsive: true, displaylogo: false });
}

function setupRegions() {
  const sel = $("regionSelect");
  const groups = {};
  for (const r of state.regions.regions) (groups[r.group] ||= []).push(r);
  for (const [g, rs] of Object.entries(groups)) {
    const og = document.createElement("optgroup");
    og.label = g;
    for (const r of rs) {
      const o = new Option(r.name, r.label);
      if (state.regions.default_selection.includes(r.label)) o.selected = true;
      og.appendChild(o);
    }
    sel.appendChild(og);
  }
  for (const id of ["regionSelect", "quantity", "showBlood"]) $(id).addEventListener("change", updateRegionPlot);
  updateRegionPlot();
}

function setupControls() {
  const tb = $("timeButtons");
  state.meta.sessions.forEach((s, t) => {
    const b = document.createElement("button");
    b.className = "btn" + (t === state.t ? " on" : "");
    b.textContent = s.label;
    b.onclick = () => {
      state.t = t;
      [...tb.children].forEach((c, k) => c.classList.toggle("on", k === t));
      drawAll();
    };
    tb.appendChild(b);
  });
  $("mode").onchange = async (e) => {
    state.mode = e.target.value;
    updateColorbar();
    if (isDti(state.mode) && !state.dti) {
      $("status").textContent = "Loading DTI…";
      await loadDti();
      $("status").textContent = state.statusText;
    }
    drawAll();
  };
  const vmax = $("vmax");
  const setV = () => { state.vmax = +vmax.value; $("vmaxVal").textContent = state.vmax.toFixed(2); updateColorbar(); drawAll(); };
  vmax.oninput = setV;
  $("alpha").oninput = (e) => { state.alpha = +e.target.value; drawAll(); };
  for (const n of Object.keys(PLANES)) $(n).addEventListener("pointerdown", (ev) => onClick(n, ev));
  setV();
  window.addEventListener("resize", drawAll);
}

async function main() {
  try {
    state.meta = await (await fetch("data/meta.json")).json();
    const m = state.meta;
    const [t1, labels, ...conc] = await Promise.all([
      loadGz("data/" + m.t1.file, Uint8Array),
      loadGz("data/" + m.labels.file, Uint16Array),
      ...m.conc.files.map((f) => loadGz("data/" + f, Uint16Array)),
    ]);
    Object.assign(state, { t1, labels, conc });
    state.regions = await (await fetch("data/regions.json")).json();
    state.cursor = m.initial_cursor;
    setupControls();
    drawAll();
    updateVoxelInfo();
    setupRegions();
    state.statusText = `Grid ${m.shape.join("×")} at ${m.spacing_mm[0]} mm. Downsampled from the Gonzo dataset (Riseth et al., CC BY 4.0).`;
    $("status").textContent = state.statusText;
    loadDti().catch((err) => console.error("DTI:", err));  // in the background, for the voxel info
  } catch (err) {
    $("status").textContent = "Could not load data: " + err.message;
    console.error(err);
  }
}
main();
