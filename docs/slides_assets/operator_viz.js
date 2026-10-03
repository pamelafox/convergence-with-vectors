// Operator visualizations for the slides, copied from web/operators.html and adapted to
// read precomputed earring + modem similarities (earring_modem.js) instead of calling the API.
(() => {
  const SVG_NS = "http://www.w3.org/2000/svg";
  // Distinct from the blue, purple and green used for a, b and ĉ
  const WORD_COLORS = ["#d1495b", "#e08a00", "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7b2d26", "#c9a227", "#5f6b7a", "#ff6f3c"];
  const fmt = (x) => x.toFixed(2);
  const fmt3 = (x) => x.toFixed(3);

  function svg(tag, attrs = {}, parent = null, text = null) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
    if (text !== null) node.textContent = text;
    if (parent) parent.appendChild(node);
    return node;
  }

  function scoreFn(op, norm) {
    switch (op) {
      case "centroid":
        return (sa, sb) => (sa + sb) / norm;
      case "maximin":
        return (sa, sb) => Math.min(sa, sb);
      case "product":
        return (sa, sb) => Math.max(sa, 0) * Math.max(sb, 0);
    }
  }

  function computeState(data) {
    const words = data.words;
    const simA = data.sim_a;
    const simB = data.sim_b;
    const cosAB = data.cos_ab;
    const idxA = words.indexOf(data.word_a);
    const idxB = words.indexOf(data.word_b);
    const norm = Math.sqrt(2 + 2 * cosAB);
    const half = Math.acos(Math.max(-1, Math.min(1, cosAB))) / 2;
    const rank = (op) => {
      const score = scoreFn(op, norm);
      return words
        .map((candidate, i) => ({ candidate, simA: simA[i], simB: simB[i], score: score(simA[i], simB[i]), i }))
        .filter((c) => c.i !== idxA && c.i !== idxB)
        .sort((x, y) => y.score - x.score || (x.candidate < y.candidate ? -1 : 1))
        .map((c, r) => ({ ...c, rank: r + 1 }));
    };
    const productAll = rank("product");
    return {
      data: { words },
      wordA: data.word_a,
      wordB: data.word_b,
      idxA,
      idxB,
      cosAB,
      norm,
      simA,
      simB,
      productAll,
      top: {
        centroid: rank("centroid").slice(0, 10),
        maximin: rank("maximin").slice(0, 10),
        product: productAll.slice(0, 10),
        textual: data.textual_top.map((c, r) => ({ ...c, rank: r + 1 })),
      },
      half,
      sh: Math.max(Math.sin(half), 1e-6),
      ch: Math.max(Math.cos(half), 1e-6),
    };
  }

  const state = computeState(window.OPERATOR_DATA);

  /** Place a word on the unit dome: x sideways and y along ĉ in the a–b plane, h = length of the hidden part. */
  function toSphere(sa, sb) {
    const x = (sb - sa) / (2 * state.sh);
    const y = (sa + sb) / (2 * state.ch);
    const r2 = x * x + y * y;
    if (r2 > 1 + 1e-6) return null;
    return { x, y, h: Math.sqrt(Math.max(0, 1 - r2)) };
  }

  function heatColor(score) {
    const t = Math.pow(Math.min(Math.max(score, 0), 1), 0.8);
    return [255 + (23 - 255) * t, 255 + (102 - 255) * t, 255 + (160 - 255) * t];
  }

  async function loadThree() {
    const THREE = await import("three");
    const { OrbitControls } = await import("three/addons/controls/OrbitControls.js");
    const { CSS2DRenderer, CSS2DObject } = await import("three/addons/renderers/CSS2DRenderer.js");
    const { Line2 } = await import("three/addons/lines/Line2.js");
    const { LineMaterial } = await import("three/addons/lines/LineMaterial.js");
    const { LineGeometry } = await import("three/addons/lines/LineGeometry.js");
    return { THREE, OrbitControls, CSS2DRenderer, CSS2DObject, Line2, LineMaterial, LineGeometry };
  }

  function createDome(container, three) {
    const { THREE, OrbitControls, CSS2DRenderer } = three;
    // Transparent background so the slide shows through
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);
    const labelRenderer = new CSS2DRenderer();
    labelRenderer.domElement.className = "dome-labels";
    container.appendChild(labelRenderer.domElement);
    const tooltip = document.createElement("div");
    tooltip.className = "dome-tooltip";
    tooltip.hidden = true;
    container.appendChild(tooltip);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(38, 4 / 3, 0.01, 100);
    camera.position.set(1.6, 1.6, -2.2);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0.35, 0);
    controls.update();
    const group = new THREE.Group();
    scene.add(group);

    const render = () => {
      renderer.render(scene, camera);
      labelRenderer.render(scene, camera);
    };
    const resolution = new THREE.Vector2(1, 1);
    const resize = () => {
      // Hidden slides have no size; the observer fires again once the slide is shown
      const w = container.clientWidth;
      const h = container.clientHeight;
      if (!w || !h) return;
      renderer.setSize(w, h);
      labelRenderer.setSize(w, h);
      resolution.set(w, h);
      group.traverse((obj) => {
        if (obj.material && obj.material.isLineMaterial) obj.material.resolution.set(w, h);
      });
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      render();
    };
    controls.addEventListener("change", render);
    new ResizeObserver(resize).observe(container);

    const raycaster = new THREE.Raycaster();
    raycaster.params.Points.threshold = 0.015;
    const dome = { ...three, resolution, group, render, points: null, pointLabels: [] };
    renderer.domElement.addEventListener("pointermove", (event) => {
      if (!dome.points) return;
      const rect = renderer.domElement.getBoundingClientRect();
      const pointer = new THREE.Vector2(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1);
      raycaster.setFromCamera(pointer, camera);
      const hit = raycaster.intersectObject(dome.points)[0];
      tooltip.hidden = !hit;
      if (hit) {
        // Tooltip is positioned inside the scaled slide, so undo reveal.js's scale
        const scale = rect.width / container.clientWidth;
        tooltip.textContent = dome.pointLabels[hit.index];
        tooltip.style.left = `${(event.clientX - rect.left) / scale + 12}px`;
        tooltip.style.top = `${(event.clientY - rect.top) / scale + 12}px`;
      }
    });

    resize();
    return dome;
  }

  function insideBoth(circleAngle) {
    const threshold = Math.cos((circleAngle * Math.PI) / 180);
    const { data, simA, simB, idxA, idxB } = state;
    const words = [];
    for (let i = 0; i < data.words.length; i++) {
      const s = Math.min(simA[i], simB[i]);
      if (i !== idxA && i !== idxB && s >= threshold - 1e-9) words.push({ candidate: data.words[i], simA: simA[i], simB: simB[i], score: s });
    }
    words.sort((x, y) => y.score - x.score || (x.candidate < y.candidate ? -1 : 1));
    // Beyond the color palette, extra words are shown as unlabeled glowing dots
    return words.slice(0, WORD_COLORS.length).map((w, i) => ({ ...w, rank: i + 1 }));
  }

  function renderDome(op, dome, ctl) {
    const topN = op === "centroid" ? state.top.centroid.slice(0, ctl.topN) : insideBoth(ctl.circleAngle);
    const { THREE, CSS2DObject, Line2, LineMaterial, LineGeometry, group } = dome;
    // WebGL lines are always 1px, so use Line2 for a width hierarchy (widths in pixels)
    const thickLine = (points, colorHex, width, dashed = false) => {
      const geometry = new LineGeometry();
      geometry.setPositions(points.flatMap((pt) => [pt.x, pt.y, pt.z]));
      const material = new LineMaterial({ color: colorHex, linewidth: width, dashed, dashSize: 0.025, gapSize: 0.02, depthTest: false, transparent: true });
      material.resolution.copy(dome.resolution);
      const line = new Line2(geometry, material);
      if (dashed) line.computeLineDistances();
      line.renderOrder = 3;
      group.add(line);
      return line;
    };
    const shaftedArrow = (dir, colorHex, width, headLength, headWidth) => {
      const helper = new THREE.ArrowHelper(dir, new THREE.Vector3(0, 0, 0), 1, colorHex, headLength, headWidth);
      helper.line.visible = false;
      helper.cone.material.depthTest = false;
      helper.cone.renderOrder = 3;
      group.add(helper);
      thickLine([new THREE.Vector3(0, 0, 0), dir.clone().multiplyScalar(1 - headLength)], colorHex, width);
    };
    const { data, simA, simB, idxA, idxB, cosAB, norm, sh, ch, wordA, wordB } = state;
    const score = scoreFn(op, norm);
    group.traverse((obj) => {
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) obj.material.dispose();
    });
    group.clear();
    // Three.js is y-up, so the hidden part h is up and ĉ points away from the camera.
    const v3 = (p, lift = 1) => new THREE.Vector3(p.x * lift, p.h * lift, -p.y * lift);

    const domeGeo = new THREE.SphereGeometry(1, 96, 48, 0, Math.PI * 2, 0, Math.PI / 2);
    const pos = domeGeo.attributes.position;
    const colors = [];
    const color = new THREE.Color();
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i);
      const y = -pos.getZ(i);
      const [r, g, b] = heatColor(score(-sh * x + ch * y, sh * x + ch * y));
      color.setRGB(r / 255, g / 255, b / 255, THREE.SRGBColorSpace);
      colors.push(color.r, color.g, color.b);
    }
    domeGeo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    group.add(new THREE.Mesh(domeGeo, new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.4, side: THREE.DoubleSide, depthWrite: false })));

    const rim = [];
    for (let i = 0; i <= 128; i++) rim.push(new THREE.Vector3(Math.cos((i / 128) * Math.PI * 2), 0, Math.sin((i / 128) * Math.PI * 2)));
    group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(rim), new THREE.LineBasicMaterial({ color: "#999" })));

    const addLabel = (text, position, colorHex, bold = false) => {
      const div = document.createElement("div");
      div.className = "dome-label";
      div.textContent = text;
      div.style.color = colorHex;
      if (bold) div.classList.add("bold");
      const label = new CSS2DObject(div);
      label.position.copy(position);
      label.center.set(-0.1, 1.1);
      group.add(label);
    };

    const topWords = new Set(topN.map((c) => c.candidate));
    const positions = [];
    const pointLabels = [];
    for (let i = 0; i < data.words.length; i++) {
      if (i === idxA || i === idxB || topWords.has(data.words[i])) continue;
      const p = toSphere(simA[i], simB[i]);
      if (!p) continue;
      const v = v3(p);
      positions.push(v.x, v.y, v.z);
      pointLabels.push(`${data.words[i]}: score ${fmt3(score(simA[i], simB[i]))}`);
    }
    const pointsGeo = new THREE.BufferGeometry();
    pointsGeo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    dome.points = new THREE.Points(pointsGeo, new THREE.PointsMaterial({ color: "#444", size: 0.018, transparent: true, opacity: 0.45 }));
    dome.pointLabels = pointLabels;
    group.add(dome.points);

    if (op === "maximin") {
      const alpha = (ctl.circleAngle * Math.PI) / 180;
      const threshold = Math.cos(alpha);
      const inside = [];
      for (let i = 0; i < data.words.length; i++) {
        if (i === idxA || i === idxB || topWords.has(data.words[i])) continue;
        const s = Math.min(simA[i], simB[i]);
        if (s < threshold - 1e-9) continue;
        const p = toSphere(simA[i], simB[i]);
        if (p) inside.push({ v: v3(p), word: data.words[i] });
      }
      if (inside.length) {
        // Categorical colors hashed from the word, so color doesn't imply closeness and stays stable as the circles move
        const insidePositions = [];
        const insideColors = [];
        inside.forEach(({ v, word }) => {
          insidePositions.push(v.x, v.y, v.z);
          let hash = 2166136261;
          for (const letter of word) hash = Math.imul(hash ^ letter.charCodeAt(0), 16777619);
          const c = new THREE.Color().setHSL((hash >>> 0) / 4294967296, 0.75, 0.5, THREE.SRGBColorSpace);
          insideColors.push(c.r, c.g, c.b);
        });
        const insideGeo = new THREE.BufferGeometry();
        insideGeo.setAttribute("position", new THREE.Float32BufferAttribute(insidePositions, 3));
        insideGeo.setAttribute("color", new THREE.Float32BufferAttribute(insideColors, 3));
        group.add(new THREE.Points(insideGeo, new THREE.PointsMaterial({ vertexColors: true, size: 0.032 })));
      }
      for (const [p, colorHex] of [
        [toSphere(1, cosAB), "#1766a0"],
        [toSphere(cosAB, 1), "#9b467d"],
      ]) {
        // Circle of points at angle alpha from the input, clipped to the dome (h >= 0)
        const center = v3(p).normalize();
        const e1 = new THREE.Vector3().crossVectors(center, new THREE.Vector3(0, 1, 0)).normalize();
        const e2 = new THREE.Vector3().crossVectors(center, e1).normalize();
        const runs = [];
        let run = [];
        for (let k = 0; k <= 180; k++) {
          const t = (k / 180) * Math.PI * 2;
          const point = center
            .clone()
            .multiplyScalar(Math.cos(alpha))
            .add(e1.clone().multiplyScalar(Math.sin(alpha) * Math.cos(t)))
            .add(e2.clone().multiplyScalar(Math.sin(alpha) * Math.sin(t)));
          if (point.y >= -1e-6) {
            run.push(point.multiplyScalar(1.004));
          } else if (run.length) {
            runs.push(run);
            run = [];
          }
        }
        if (run.length) runs.push(run);
        for (const r of runs) thickLine(r, colorHex, 3);
      }
    }

    // Protractor-style arc between two unit vectors, slerped so it stays in their plane
    const drawArc = (from, to, colorHex, radius, width, fillOpacity) => {
      const theta = from.angleTo(to);
      const arcPoints = [];
      for (let k = 0; k <= 48; k++) {
        const t = k / 48;
        const point = from
          .clone()
          .multiplyScalar(Math.sin((1 - t) * theta))
          .add(to.clone().multiplyScalar(Math.sin(t * theta)));
        arcPoints.push(point.divideScalar(Math.sin(theta)).multiplyScalar(radius));
      }
      // Shaded wedge: a triangle fan from the origin to the arc
      const wedge = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), ...arcPoints]);
      wedge.setIndex(arcPoints.slice(1).flatMap((_, k) => [0, k + 1, k + 2]));
      const wedgeMesh = new THREE.Mesh(wedge, new THREE.MeshBasicMaterial({ color: colorHex, transparent: true, opacity: fillOpacity, side: THREE.DoubleSide, depthTest: false, depthWrite: false }));
      wedgeMesh.renderOrder = 2;
      group.add(wedgeMesh);
      thickLine(arcPoints, colorHex, width);
      return { theta, arcPoints };
    };
    const vecA = v3(toSphere(1, cosAB)).normalize();
    const vecB = v3(toSphere(cosAB, 1)).normalize();
    for (const c of topN) {
      const p = toSphere(c.simA, c.simB);
      if (!p) continue;
      const isWinner = c.rank === 1;
      const wordColor = WORD_COLORS[(c.rank - 1) % WORD_COLORS.length];
      const ball = new THREE.Mesh(new THREE.SphereGeometry(isWinner ? 0.028 : 0.018, 16, 12), new THREE.MeshBasicMaterial({ color: wordColor }));
      ball.position.copy(v3(p));
      group.add(ball);
      shaftedArrow(v3(p).normalize(), wordColor, isWinner ? 3.5 : 2.5, 0.05, 0.025);
      // Drop line: the hidden part, ending at the word's spot on the a–b plane
      thickLine([v3(p), v3({ ...p, h: 0 })], wordColor, 1.5, true);
      addLabel(`${c.rank}. ${c.candidate}`, v3(p), wordColor, isWinner);

      if (op === "maximin") {
        // The larger angle (the weaker similarity) sets the maximin score, so it gets the thick arc
        const w = v3(p).normalize();
        const angles = [
          { from: vecA, theta: vecA.angleTo(w) },
          { from: vecB, theta: vecB.angleTo(w) },
        ];
        const worst = Math.max(angles[0].theta, angles[1].theta);
        for (const { from, theta } of angles) {
          const isWorst = theta === worst;
          drawArc(from, w, wordColor, 0.5, isWorst ? 5 : 1.5, isWorst ? 0.3 : 0.08);
        }
      } else {
        const { theta, arcPoints } = drawArc(new THREE.Vector3(0, 0, -1), v3(p).normalize(), wordColor, 0.4, isWinner ? 3.5 : 2.5, isWinner ? 0.3 : 0.12);
        addLabel(`${Math.round((theta * 180) / Math.PI)}°`, arcPoints[40], wordColor, isWinner);
      }
    }

    const arrow = (p, colorHex, text) => {
      shaftedArrow(v3(p).normalize(), colorHex, 4, 0.07, 0.04);
      addLabel(text, v3(p), colorHex, true);
    };
    arrow(toSphere(1, cosAB), "#1766a0", `a "${wordA}"`);
    arrow(toSphere(cosAB, 1), "#9b467d", `b "${wordB}"`);
    // Maximin never uses the midpoint, so only centroid draws ĉ
    if (op === "centroid") arrow({ x: 0, y: 1, h: 0 }, "#2a7d46", "ĉ");
    dome.render();
  }

  /** Every word as a dot at (s_a, s_b), with each top word's s_a × s_b rectangle and the winner's equal-product curve. */
  function renderProduct(root, topN) {
    root.replaceChildren();
    const ox = 52;
    const oy = 262;
    const S = 240;
    root.setAttribute("viewBox", "0 0 330 300");
    const x = (s) => ox + Math.max(s, 0) * S;
    const y = (s) => oy - Math.max(s, 0) * S;
    const { data, simA, simB, idxA, idxB, wordA, wordB } = state;

    const tick = { "font-size": 13, fill: "#666" };
    svg("line", { x1: ox, y1: oy, x2: ox + S, y2: oy, stroke: "#999" }, root);
    svg("line", { x1: ox, y1: oy, x2: ox, y2: oy - S, stroke: "#999" }, root);
    svg("text", { ...tick, x: ox - 6, y: oy + 14, "text-anchor": "end" }, root, "0");
    svg("text", { ...tick, x: ox + S, y: oy + 16, "text-anchor": "middle" }, root, "1");
    svg("text", { ...tick, x: ox - 6, y: oy - S + 5, "text-anchor": "end" }, root, "1");
    svg("text", { x: ox + S / 2, y: oy + 30, "text-anchor": "middle", "font-size": 14, fill: "#1766a0" }, root, `s_a: "${wordA}" →`);
    svg("text", { x: ox - 22, y: oy - S / 2, "text-anchor": "middle", "font-size": 14, fill: "#9b467d", transform: `rotate(-90 ${ox - 22} ${oy - S / 2})` }, root, `s_b: "${wordB}" →`);

    for (let i = 0; i < data.words.length; i++) {
      if (i === idxA || i === idxB) continue;
      svg("circle", { cx: x(simA[i]), cy: y(simB[i]), r: 1.4, fill: "#444", "fill-opacity": 0.25 }, root);
    }

    // Every point on this curve has the same product as the winner, so nothing lies beyond it
    const best = Math.max(topN[0].simA, 0) * Math.max(topN[0].simB, 0);
    const curve = [];
    for (let k = 0; k <= 40; k++) {
      const sa = best + ((1 - best) * k) / 40;
      curve.push(`${k ? "L" : "M"}${x(sa).toFixed(1)},${y(best / sa).toFixed(1)}`);
    }
    svg("path", { d: curve.join(" "), fill: "none", stroke: "#222", "stroke-width": 1.2, "stroke-dasharray": "4 3" }, root);
    svg("text", { x: x(best) + 8, y: y(0.95), "font-size": 13, fill: "#222" }, root, `s_a × s_b = ${fmt(best)}`);

    for (const c of [...topN].reverse()) {
      const color = WORD_COLORS[(c.rank - 1) % WORD_COLORS.length];
      const isWinner = c.rank === 1;
      svg("rect", { x: ox, y: y(c.simB), width: x(c.simA) - ox, height: oy - y(c.simB), fill: color, "fill-opacity": isWinner ? 0.2 : 0.06, stroke: color, "stroke-width": isWinner ? 2.5 : 1.2 }, root);
      svg("circle", { cx: x(c.simA), cy: y(c.simB), r: isWinner ? 4.5 : 3, fill: color }, root);
      svg("text", { x: x(c.simA) + 6, y: y(c.simB) - 5, "font-size": isWinner ? 15 : 12, "font-weight": isWinner ? 700 : 500, fill: color }, root, `${c.rank}. ${c.candidate}`);
    }
  }

  function renderSums(table, words, op) {
    const cell = (tag, text, color, bold = false) => {
      const el = document.createElement(tag);
      el.textContent = text;
      if (color) el.style.color = color;
      if (bold) el.style.fontWeight = "700";
      return el;
    };
    const isMaximin = op === "maximin";
    const isProduct = op === "product";
    const head = document.createElement("tr");
    head.append(cell("th", "word"), cell("th", "s_a"), cell("th", "s_b"), cell("th", isMaximin ? "min" : isProduct ? "product" : "sum"));
    const combine = (c) => (isMaximin ? Math.min(c.simA, c.simB) : isProduct ? Math.max(c.simA, 0) * Math.max(c.simB, 0) : c.simA + c.simB);
    const rows = words.map((c) => {
      const tr = document.createElement("tr");
      // For maximin, bold the weaker similarity since it's the score
      tr.append(
        cell("td", `${c.rank}. ${c.candidate}`, WORD_COLORS[(c.rank - 1) % WORD_COLORS.length]),
        cell("td", fmt(c.simA), null, isMaximin && c.simA <= c.simB),
        cell("td", fmt(c.simB), null, isMaximin && c.simB < c.simA),
        cell("td", fmt(combine(c))),
      );
      return tr;
    });
    const thead = document.createElement("thead");
    thead.appendChild(head);
    const tbody = document.createElement("tbody");
    tbody.replaceChildren(...rows);
    table.replaceChildren(thead, tbody);
  }

  function renderComparison(table, ops, rows) {
    const headRow = document.createElement("tr");
    for (const label of ["rank", ...ops]) {
      const th = document.createElement("th");
      th.textContent = label.replace("_", " ");
      headRow.appendChild(th);
    }
    const body = [];
    // One color per word from ColorBrewer's Set3, a palette designed so each color is distinct
    const PALETTE = ["#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3", "#fdb462", "#b3de69", "#fccde5", "#d9d9d9", "#bc80bd", "#ccebc5", "#ffed6f"];
    const colors = new Map();
    for (let r = 0; r < rows; r++) {
      for (const op of ops) {
        const c = state.top[op][r];
        if (c && !colors.has(c.candidate)) colors.set(c.candidate, PALETTE[colors.size % PALETTE.length]);
      }
    }
    for (let r = 0; r < rows; r++) {
      const tr = document.createElement("tr");
      const rankCell = document.createElement("td");
      rankCell.textContent = `#${r + 1}`;
      tr.appendChild(rankCell);
      for (const op of ops) {
        const td = document.createElement("td");
        const c = state.top[op][r];
        if (c) {
          td.textContent = c.candidate;
          td.style.background = colors.get(c.candidate);
        }
        tr.appendChild(td);
      }
      body.push(tr);
    }
    const thead = document.createElement("thead");
    thead.appendChild(headRow);
    const tbody = document.createElement("tbody");
    tbody.replaceChildren(...body);
    table.replaceChildren(thead, tbody);
  }

  async function init() {
    for (const root of document.querySelectorAll(".viz-product")) {
      const section = root.closest("section");
      const slider = section.querySelector(".viz-topn");
      const out = section.querySelector(".viz-topn-out");
      const sums = section.querySelector(".viz-sums");
      const update = () => {
        const n = parseInt(slider.value, 10);
        out.textContent = n === 1 ? "word" : `${n} words`;
        const topN = state.productAll.slice(0, n);
        renderProduct(root, topN);
        renderSums(sums, topN, "product");
      };
      slider.addEventListener("input", update);
      update();
    }
    for (const table of document.querySelectorAll(".viz-compare")) {
      renderComparison(table, ["centroid", "maximin", "product", "textual"], 5);
    }

    const three = await loadThree();
    for (const container of document.querySelectorAll(".viz-dome")) {
      const op = container.dataset.op;
      const dome = createDome(container, three);
      const ctl = { topN: 3, circleAngle: (Math.acos(Math.min(1, state.top.maximin[0].score)) * 180) / Math.PI };
      const section = container.closest("section");
      const topNSlider = section.querySelector(".viz-topn");
      if (topNSlider) {
        const out = section.querySelector(".viz-topn-out");
        const sums = section.querySelector(".viz-sums");
        const update = () => {
          ctl.topN = parseInt(topNSlider.value, 10);
          if (out) out.textContent = ctl.topN === 1 ? "word" : `${ctl.topN} words`;
          renderDome(op, dome, ctl);
          if (sums) renderSums(sums, state.top.centroid.slice(0, ctl.topN), op);
        };
        topNSlider.addEventListener("input", update);
        update();
        continue;
      }
      const slider = section.querySelector(".viz-circle");
      if (slider) {
        slider.value = ctl.circleAngle;
        const out = section.querySelector(".viz-circle-out");
        const simOut = section.querySelector(".viz-circle-sim");
        const sums = section.querySelector(".viz-sums");
        const update = () => {
          ctl.circleAngle = parseFloat(slider.value);
          if (out) out.textContent = ctl.circleAngle.toFixed(1);
          if (simOut) simOut.textContent = fmt(Math.cos((ctl.circleAngle * Math.PI) / 180));
          renderDome(op, dome, ctl);
          if (sums) renderSums(sums, insideBoth(ctl.circleAngle), op);
        };
        slider.addEventListener("input", update);
        update();
      } else {
        renderDome(op, dome, ctl);
      }
    }
  }

  init().catch((err) => {
    for (const container of document.querySelectorAll(".viz-dome")) container.textContent = `Couldn't load the 3D view: ${err.message}`;
  });
})();
