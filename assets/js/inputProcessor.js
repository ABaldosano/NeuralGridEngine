"use strict";

/**
 * InputProcessor
 * Converts raw canvas pixel data into a normalized vector suitable
 * for the network's input layer. Isolated so preprocessing can be
 * swapped later without touching the network or UI code.
 */
const InputProcessor = {
  canvasToInputVector(canvas, targetSize = 28) {
    if (!(canvas instanceof HTMLCanvasElement)) {
      throw new TypeError("InputProcessor.canvasToInputVector: canvas must be an HTMLCanvasElement");
    }
    if (!isInteger(targetSize) || targetSize <= 0) {
      throw new RangeError("InputProcessor.canvasToInputVector: targetSize must be a positive integer");
    }

    const offscreen = document.createElement("canvas");
    offscreen.width = targetSize;
    offscreen.height = targetSize;
    const offscreenContext = offscreen.getContext("2d");

    offscreenContext.drawImage(canvas, 0, 0, targetSize, targetSize);

    const imageData = offscreenContext.getImageData(0, 0, targetSize, targetSize);
    const rawPixels = [];
    for (let i = 0; i < imageData.data.length; i += 4) {
      rawPixels.push(imageData.data[i]); // red channel; drawing is white-on-black
    }

    return normalizePixels(rawPixels);
  },
};