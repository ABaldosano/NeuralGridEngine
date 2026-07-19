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

  /** Creates an independent controller instance bound to its own canvas, so multiple draw pads can coexist on one page without sharing state. */
  create(canvas) {
    if (!(canvas instanceof HTMLCanvasElement)) {
      throw new TypeError("CanvasController.create: canvas must be an HTMLCanvasElement");
    }
    const controller = Object.create(CanvasController);
    controller.canvas = canvas;
    controller.ctx = canvas.getContext("2d");
    controller.drawing = false;
    controller.clear();

    canvas.addEventListener("mousedown", (e) => controller.start(e));
    canvas.addEventListener("mousemove", (e) => controller.move(e));
    window.addEventListener("mouseup", () => controller.end());

    canvas.addEventListener("touchstart", (e) => controller.start(e), { passive: false });
    canvas.addEventListener("touchmove", (e) => controller.move(e), { passive: false });
    window.addEventListener("touchend", () => controller.end());

    return controller;
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
    const wasDrawing = this.drawing;
    this.drawing = false;
    // Only fires for a stroke that was actually happening on *this*
    // canvas — window-level mouseup/touchend listeners catch every
    // release on the page (e.g. clicking the Clear button), and we
    // don't want those mistaken for "the user just finished drawing".
    if (wasDrawing && typeof this.onStrokeEnd === "function") this.onStrokeEnd();
  },

  clear() {
    this.ctx.fillStyle = "#000";
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
  },
};