"use strict";

/**
 * Preprocessing
 * Converts a raw drawing canvas into a clean, centered, normalized
 * 28x28 input vector. Used identically by the production recognizer
 * and the training lab so both see the same input distribution.
 */
const Preprocessing = {
  /**
   * @param {HTMLCanvasElement} sourceCanvas
   * @param {number} [targetSize=28]
   * @returns {number[]} normalized 0..1 input vector, length targetSize^2
   */
  canvasToInputVector(sourceCanvas, targetSize = 28) {
    if (!(sourceCanvas instanceof HTMLCanvasElement)) {
      throw new TypeError("Preprocessing.canvasToInputVector: sourceCanvas must be an HTMLCanvasElement");
    }

    const raw = this._readGrayscale(sourceCanvas);
    const bbox = this._boundingBox(raw, sourceCanvas.width);

    if (!bbox) {
      return normalizePixels(new Array(targetSize * targetSize).fill(0));
    }

    const cropped = this._crop(raw, sourceCanvas.width, bbox);
    const centered = this._centerOnCanvas(cropped, targetSize);
    return normalizePixels(centered);
  },

  /** Reads the red channel (drawing is white-on-black) into a flat array. */
  _readGrayscale(canvas) {
    const ctx = canvas.getContext("2d");
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const pixels = new Array(canvas.width * canvas.height);
    for (let i = 0, p = 0; i < imageData.data.length; i += 4, p++) {
      pixels[p] = imageData.data[i];
    }
    return pixels;
  },

  /** Finds the tight bounding box of non-background ink, with a threshold. */
  _boundingBox(pixels, width, threshold = 20) {
    const height = pixels.length / width;
    let minX = width, maxX = -1, minY = height, maxY = -1;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (pixels[y * width + x] > threshold) {
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }

    if (maxX < 0) return null;
    return { minX, minY, maxX, maxY };
  },

  /** Crops the pixel buffer to the bounding box, returns {data,width,height}. */
  _crop(pixels, sourceWidth, bbox) {
    const width = bbox.maxX - bbox.minX + 1;
    const height = bbox.maxY - bbox.minY + 1;
    const data = new Array(width * height);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        data[y * width + x] = pixels[(bbox.minY + y) * sourceWidth + (bbox.minX + x)];
      }
    }
    return { data, width, height };
  },

  /**
   * Scales the cropped glyph to fit within a 20/28 inner box (preserving
   * aspect ratio, matching common digit-recognition conventions), then
   * centers it on a targetSize canvas using center-of-mass alignment.
   */
  _centerOnCanvas(cropped, targetSize) {
    const innerSize = Math.round(targetSize * (20 / 28));
    const scale = Math.min(innerSize / cropped.width, innerSize / cropped.height);
    const scaledWidth = Math.max(1, Math.round(cropped.width * scale));
    const scaledHeight = Math.max(1, Math.round(cropped.height * scale));

    const scaledCanvas = document.createElement("canvas");
    scaledCanvas.width = scaledWidth;
    scaledCanvas.height = scaledHeight;
    const scaledCtx = scaledCanvas.getContext("2d");

    const sourceCanvas = document.createElement("canvas");
    sourceCanvas.width = cropped.width;
    sourceCanvas.height = cropped.height;
    const sourceCtx = sourceCanvas.getContext("2d");
    const sourceImageData = sourceCtx.createImageData(cropped.width, cropped.height);
    for (let i = 0; i < cropped.data.length; i++) {
      const v = cropped.data[i];
      sourceImageData.data[i * 4] = v;
      sourceImageData.data[i * 4 + 1] = v;
      sourceImageData.data[i * 4 + 2] = v;
      sourceImageData.data[i * 4 + 3] = 255;
    }
    sourceCtx.putImageData(sourceImageData, 0, 0);

    scaledCtx.imageSmoothingEnabled = true;
    scaledCtx.drawImage(sourceCanvas, 0, 0, scaledWidth, scaledHeight);
    const scaledData = scaledCtx.getImageData(0, 0, scaledWidth, scaledHeight);
    const scaledPixels = new Array(scaledWidth * scaledHeight);
    for (let i = 0, p = 0; i < scaledData.data.length; i += 4, p++) {
      scaledPixels[p] = scaledData.data[i];
    }

    let sumX = 0, sumY = 0, sumMass = 0;
    for (let y = 0; y < scaledHeight; y++) {
      for (let x = 0; x < scaledWidth; x++) {
        const mass = scaledPixels[y * scaledWidth + x];
        sumX += x * mass;
        sumY += y * mass;
        sumMass += mass;
      }
    }
    const centerOfMassX = sumMass > 0 ? sumX / sumMass : scaledWidth / 2;
    const centerOfMassY = sumMass > 0 ? sumY / sumMass : scaledHeight / 2;

    const offsetX = Math.round(targetSize / 2 - centerOfMassX);
    const offsetY = Math.round(targetSize / 2 - centerOfMassY);

    const final = new Array(targetSize * targetSize).fill(0);
    for (let y = 0; y < scaledHeight; y++) {
      const ty = y + offsetY;
      if (ty < 0 || ty >= targetSize) continue;
      for (let x = 0; x < scaledWidth; x++) {
        const tx = x + offsetX;
        if (tx < 0 || tx >= targetSize) continue;
        final[ty * targetSize + tx] = scaledPixels[y * scaledWidth + x];
      }
    }
    return final;
  },
};