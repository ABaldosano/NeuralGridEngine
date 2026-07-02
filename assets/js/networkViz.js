"use strict";

/**
 * NetworkViz
 * Real-time pseudo-3D visualization of the recognizer network, drawn
 * with the project's existing Canvas 2D rendering (no new deps).
 * Layers are laid out as depth-shifted vertical columns (isometric-
 * style offsets) with a slow continuous parallax animation, glowing
 * nodes colored by activation strength, and smooth interpolation
 * between activation frames. The color legend lives in the DOM
 * (see index.html / .network-legend) next to this canvas.
 */
const NetworkViz = {
  canvas: null,
  ctx: null,
  time: 0,
  rafId: null,

  // Current (displayed, smoothly-interpolated) and target activation
  // frames. Each is { input:number[], hidden1:number[], hidden2:number[], output:number[] }.
  current: null,
  target: null,
  predictedIndex: null,

  LAYER_LABELS: [null, null, null, "0123456789"],

  init(canvas) {
    if (!(canvas instanceof HTMLCanvasElement)) {
      throw new TypeError("NetworkViz.init: canvas must be an HTMLCanvasElement");
    }
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");

    const empty = {
      input: new Array(28).fill(0),
      hidden1: new Array(24).fill(0),
      hidden2: new Array(18).fill(0),
      output: new Array(10).fill(0),
    };
    this.current = this._cloneFrame(empty);
    this.target = this._cloneFrame(empty);
    this.predictedIndex = null;

    if (!this.rafId) this._loop();
  },

  _cloneFrame(frame) {
    return {
      input: [...frame.input],
      hidden1: [...frame.hidden1],
      hidden2: [...frame.hidden2],
      output: [...frame.output],
    };
  },

  /** Maps an activation (0..1 normalized) to a spec-defined color. */
  colorFor(normalizedValue) {
    const v = clamp(normalizedValue, 0, 1);
    if (v <= 0.02) return { r: 0x20, g: 0x20, b: 0x20 };
    if (v < 0.25) return { r: 0x3b, g: 0x82, b: 0xf6 };
    if (v < 0.5) return { r: 0x22, g: 0xc5, b: 0x5e };
    if (v < 0.75) return { r: 0xfa, g: 0xcc, b: 0x15 };
    return { r: 0xef, g: 0x44, b: 0x44 };
  },

  _rgb(c) {
    return `rgb(${c.r},${c.g},${c.b})`;
  },

  /**
   * Called by app.js after a prediction. Stores new target activations;
   * the render loop smoothly interpolates toward them every frame.
   */
  render(inputVector, activations, predictedIndex = null) {
    const inputSample = this.sample(inputVector, 28);
    const hidden1Sample = this.sample(activations[0].output || new Array(activations[0].size).fill(0), 24);
    const hidden2Sample = this.sample(activations[1].output || new Array(activations[1].size).fill(0), 18);
    const output = activations[2].output || new Array(10).fill(0);

    this.target = {
      input: inputSample,
      hidden1: this._normalizeLayer(hidden1Sample),
      hidden2: this._normalizeLayer(hidden2Sample),
      output: this._normalizeLayer(output),
    };
    this.predictedIndex = predictedIndex;
  },

  /** ReLU outputs are unbounded; scale each layer relative to its own max so colors stay meaningful. */
  _normalizeLayer(values) {
    const peak = Math.max(...values, 0.0001);
    return values.map((v) => clamp(v / peak, 0, 1));
  },

  sample(array, count) {
    if (array.length <= count) return [...array];
    const step = array.length / count;
    const result = [];
    for (let i = 0; i < count; i++) result.push(array[Math.floor(i * step)]);
    return result;
  },

  /** Continuous animation loop: lerps current -> target and redraws. */
  _loop() {
    this.rafId = requestAnimationFrame(() => this._loop());
    if (!this.ctx) return;

    this.time += 0.016;
    const lerpFactor = 0.15;
    ["input", "hidden1", "hidden2", "output"].forEach((key) => {
      this.current[key] = this.current[key].map((value, i) => lerp(value, this.target[key][i], lerpFactor));
    });

    this._draw();
  },

  _draw() {
    const ctx = this.ctx;
    const canvas = this.canvas;
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, width, height);

    const layers = [
      { nodes: this.current.input, raw: this.target.input, isInput: true },
      { nodes: this.current.hidden1, raw: this.target.hidden1 },
      { nodes: this.current.hidden2, raw: this.target.hidden2 },
      { nodes: this.current.output, raw: this.target.output, labeled: true },
    ];

    // Slow pseudo-3D parallax: each column drifts slightly on a
    // sine wave, offset by depth, to read as a gently rotating scene.
    const depthShiftX = 48;
    const colGap = (width - 200) / (layers.length - 1);
    const positions = [];

    layers.forEach((layer, li) => {
      const wave = Math.sin(this.time * 0.6 + li * 0.8) * 8;
      const baseX = 100 + li * colGap - li * depthShiftX + wave;
      const n = layer.nodes.length;
      const rowGap = (height - 90) / Math.max(n - 1, 1);
      const colPositions = [];
      for (let i = 0; i < n; i++) {
        colPositions.push({ x: baseX, y: 45 + i * rowGap + li * 8 });
      }
      positions.push(colPositions);
    });

    // Connections between adjacent layers: thin lines whose color and
    // brightness reflect the activation flowing from source to target,
    // plus a traveling pulse so the light reads as moving along the wire.
    for (let li = 0; li < positions.length - 1; li++) {
      const a = positions[li];
      const b = positions[li + 1];
      const aVals = layers[li].nodes;
      const bVals = layers[li + 1].nodes;
      for (let i = 0; i < a.length; i++) {
        const av = aVals[i];
        for (let j = 0; j < b.length; j++) {
          const bv = bVals[j];
          const signal = Math.max(av, bv);
          const pulse = 0.5 + 0.5 * Math.sin(this.time * 3 + (i + j) * 0.35);
          const alpha = 0.03 + signal * 0.55 * pulse;

          const from = a[i];
          const to = b[j];
          const gradient = ctx.createLinearGradient(from.x, from.y, to.x, to.y);
          const sourceColor = this.colorFor(av);
          const targetColor = this.colorFor(bv);
          gradient.addColorStop(0, `rgba(${sourceColor.r},${sourceColor.g},${sourceColor.b},${alpha})`);
          gradient.addColorStop(1, `rgba(${targetColor.r},${targetColor.g},${targetColor.b},${alpha})`);

          ctx.strokeStyle = gradient;
          ctx.lineWidth = 0.4 + signal * 1.1;
          if (signal > 0.5) {
            ctx.shadowColor = this._rgb(targetColor);
            ctx.shadowBlur = signal * 4;
          }
          ctx.beginPath();
          ctx.moveTo(from.x, from.y);
          ctx.lineTo(to.x, to.y);
          ctx.stroke();
          ctx.shadowBlur = 0;
        }
      }
    }

    // Nodes: emissive glow + smooth color interpolation, numbered labels.
    layers.forEach((layer, li) => {
      const pts = positions[li];
      layer.nodes.forEach((value, i) => {
        const { x, y } = pts[i];
        const color = this.colorFor(value);
        const fill = this._rgb(color);
        const isPredicted = layer.labeled && this.predictedIndex === i;
        const radius = (isPredicted ? 13 : 10) + value * 4;
        const glowSize = (isPredicted ? 10 : 6) + value * 30;

        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fillStyle = fill;
        ctx.shadowColor = fill;
        ctx.shadowBlur = glowSize;
        ctx.fill();
        ctx.shadowBlur = 0;

        if (isPredicted) {
          const ringPulse = 3 + Math.sin(this.time * 4) * 2;
          ctx.beginPath();
          ctx.arc(x, y, radius + 6 + ringPulse, 0, Math.PI * 2);
          ctx.strokeStyle = "#fafafa";
          ctx.lineWidth = 2.5;
          ctx.shadowColor = "#fafafa";
          ctx.shadowBlur = 8;
          ctx.stroke();
          ctx.shadowBlur = 0;
        }

        if (layer.labeled) {
          ctx.fillStyle = isPredicted ? "#fafafa" : "#e5e5e5";
          ctx.font = isPredicted ? "bold 22px Inter, sans-serif" : "20px Inter, sans-serif";
          ctx.textBaseline = "middle";
          ctx.fillText(String(i), x + (isPredicted ? 26 : 22), y + 1);
        }
      });
    });

    if (this.predictedIndex !== null && this.predictedIndex !== undefined) {
      const predictedPt = positions[positions.length - 1][this.predictedIndex];
      if (predictedPt) {
        ctx.fillStyle = "#fafafa";
        ctx.font = "bold 13px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("PREDICTED", predictedPt.x, predictedPt.y - 26);
        ctx.textAlign = "left";
      }
    }
  },
};