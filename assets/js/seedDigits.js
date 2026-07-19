"use strict";

const SeedDigits = {
  PATHS: {
    0: [[[14, 4], [8, 8], [6, 14], [6, 20], [8, 24], [14, 26], [20, 24], [22, 20], [22, 14], [20, 8], [14, 4]]],
    1: [[[10, 6], [14, 4], [14, 26]], [[9, 26], [19, 26]]],
    2: [[[7, 8], [10, 4], [17, 4], [21, 8], [19, 14], [6, 24], [6, 26], [22, 26]]],
    3: [[[7, 5], [19, 5], [12, 13], [19, 15], [20, 22], [15, 26], [8, 24], [7, 21]]],
    4: [[[17, 4], [6, 18], [23, 18]], [[17, 4], [17, 26]]],
    5: [[[20, 4], [7, 4], [6, 13], [14, 13], [20, 17], [19, 24], [12, 27], [6, 24]]],
    6: [[[19, 5], [10, 8], [6, 16], [7, 23], [13, 27], [19, 24], [20, 18], [15, 14], [8, 16]]],
    7: [[[6, 5], [22, 5], [12, 26]]],
    8: [[[14, 4], [9, 7], [9, 11], [14, 14], [19, 11], [19, 7], [14, 4]], [[14, 14], [8, 18], [8, 23], [14, 27], [20, 23], [20, 18], [14, 14]]],
    9: [[[19, 12], [16, 15], [10, 14], [8, 10], [10, 5], [16, 5], [19, 9], [19, 20], [14, 27], [8, 25]]],
  },

  /**
   * Renders one jittered instance of a digit directly into a flat
   * 28x28 pixel buffer (0..255) and returns the normalized vector.
   * @param {number} digit
   * @param {number} [size=28]
   * @returns {number[]}
   */
  renderSample(digit, size = 28) {
    const pixels = new Array(size * size).fill(0);

    const dx = randomFloat(-2.6, 2.6);
    const dy = randomFloat(-2.6, 2.6);
    const scaleX = randomFloat(0.72, 1.28);
    const scaleY = randomFloat(0.72, 1.28);
    const angle = degreesToRadians(randomFloat(-18, 18));
    const thickness = randomFloat(1.3, 3.4);
    const vertexJitter = randomFloat(0, 1.1);
    const cx = size / 2;
    const cy = size / 2;

    const transform = ([x, y]) => {
      const jx = x + randomFloat(-vertexJitter, vertexJitter);
      const jy = y + randomFloat(-vertexJitter, vertexJitter);
      const px = (jx - cx) * scaleX;
      const py = (jy - cy) * scaleY;
      const rx = px * Math.cos(angle) - py * Math.sin(angle);
      const ry = px * Math.sin(angle) + py * Math.cos(angle);
      return [rx + cx + dx, ry + cy + dy];
    };

    for (const stroke of SeedDigits.PATHS[digit]) {
      for (let i = 0; i < stroke.length - 1; i++) {
        const [x0, y0] = transform(stroke[i]);
        const [x1, y1] = transform(stroke[i + 1]);
        SeedDigits._drawThickLine(pixels, size, x0, y0, x1, y1, thickness);
      }
    }

    return normalizePixels(pixels);
  },

  /** Stamps overlapping soft disks along a segment to fake stroke width. */
  _drawThickLine(pixels, size, x0, y0, x1, y1, thickness) {
    const length = Math.hypot(x1 - x0, y1 - y0);
    const steps = Math.max(Math.ceil(length * 1.5), 1);
    const radius = thickness / 2;
    for (let s = 0; s <= steps; s++) {
      const t = s / steps;
      SeedDigits._stampDisk(pixels, size, x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, radius);
    }
  },

  /** Softly stamps a small disk of "ink" centered at (cx, cy). */
  _stampDisk(pixels, size, cx, cy, radius) {
    const minX = Math.max(0, Math.floor(cx - radius));
    const maxX = Math.min(size - 1, Math.ceil(cx + radius));
    const minY = Math.max(0, Math.floor(cy - radius));
    const maxY = Math.min(size - 1, Math.ceil(cy + radius));

    for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
        const distance = Math.hypot(x - cx, y - cy);
        if (distance <= radius + 0.5) {
          const intensity = 255 * clamp(1 - distance / (radius + 0.6), 0, 1);
          const index = y * size + x;
          if (intensity > pixels[index]) pixels[index] = intensity;
        }
      }
    }
  },

  /**
   * Builds a labeled dataset covering digits 0-9 with many jittered
   * variants each, for baseline pretraining.
   * @param {number} [variantsPerDigit=220]
   * @returns {Array<{input:number[], label:number}>}
   */
  generateDataset(variantsPerDigit = 220) {
    const dataset = [];
    for (let digit = 0; digit <= 9; digit++) {
      for (let v = 0; v < variantsPerDigit; v++) {
        dataset.push({ input: SeedDigits.renderSample(digit), label: digit });
      }
    }
    return dataset;
  },
};