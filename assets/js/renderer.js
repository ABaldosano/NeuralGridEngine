"use strict";

/**
 * CanvasController
 * Handles mouse + touch drawing on the visible input canvas.
 * Purely presentational — no network or preprocessing logic here.
 */
const CanvasController = {
  init(canvas) {
    if (!(canvas instanceof HTMLCanvasElement)) {
      throw new TypeError("CanvasController.init: canvas must be an HTMLCanvasElement");
    }
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.drawing = false;

    this.clear();

    canvas.addEventListener("mousedown", (e) => this.start(e));
    canvas.addEventListener("mousemove", (e) => this.move(e));
    window.addEventListener("mouseup", () => this.end());

    canvas.addEventListener("touchstart", (e) => this.start(e), { passive: false });
    canvas.addEventListener("touchmove", (e) => this.move(e), { passive: false });
    window.addEventListener("touchend", () => this.end());
  },

  getPos(e) {
    const rect = this.canvas.getBoundingClientRect();
    const point = e.touches ? e.touches[0] : e;
    return { x: point.clientX - rect.left, y: point.clientY - rect.top };
  },

  start(e) {
    e.preventDefault();
    this.drawing = true;
    const { x, y } = this.getPos(e);
    this.ctx.beginPath();
    this.ctx.moveTo(x, y);
  },

  move(e) {
    if (!this.drawing) return;
    e.preventDefault();
    const { x, y } = this.getPos(e);
    this.ctx.lineWidth = 16;
    this.ctx.lineCap = "round";
    this.ctx.strokeStyle = "#fff";
    this.ctx.lineTo(x, y);
    this.ctx.stroke();
  },

  end() {
    this.drawing = false;
  },

  clear() {
    this.ctx.fillStyle = "#000";
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
  },
};