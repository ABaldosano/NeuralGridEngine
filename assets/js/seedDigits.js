"use strict";

/**
 * SeedDigits
 * Generates a small synthetic dataset of hand-drawn-looking digits
 * (0-9) by stroking vector paths onto an offscreen canvas with random
 * jitter, so the network has baseline knowledge of all ten digits
 * before any user supervision happens.
 */
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
   * Draws one jittered instance of a digit path onto a 28x28 canvas
   * and returns the normalized input vector.
   * @param {number} digit
   * @returns {number[]}
   */
  renderSample(digit) {
    const size = 28;
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, size, size);

    const dx = randomFloat(-1.5, 1.5);
    const dy = randomFloat(-1.5, 1.5);
    const scale = randomFloat(0.9, 1.1);
    const cx = size / 2;
    const cy = size / 2;
    const angle = degreesToRadians(randomFloat(-8, 8));

    ctx.strokeStyle = "#fff";
    ctx.lineWidth = randomFloat(2.2, 3.2);
    ctx.lineJoin = "round";
    ctx.lineCap = "round";

    const transform = ([x, y]) => {
      let px = (x - cx) * scale;
      let py = (y - cy) * scale;
      const rx = px * Math.cos(angle) - py * Math.sin(angle);
      const ry = px * Math.sin(angle) + py * Math.cos(angle);
      return [rx + cx + dx, ry + cy + dy];
    };

    for (const stroke of SeedDigits.PATHS[digit]) {
      ctx.beginPath();
      stroke.forEach((point, i) => {
        const [x, y] = transform(point);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }

    const imageData = ctx.getImageData(0, 0, size, size);
    const rawPixels = [];
    for (let i = 0; i < imageData.data.length; i += 4) {
      rawPixels.push(imageData.data[i]);
    }
    return normalizePixels(rawPixels);
  },

  /**
   * Builds a labeled dataset covering digits 0-9 with several
   * jittered variants each, for baseline pretraining.
   * @param {number} [variantsPerDigit=25]
   * @returns {Array<{input:number[], label:number}>}
   */
  generateDataset(variantsPerDigit = 25) {
    const dataset = [];
    for (let digit = 0; digit <= 9; digit++) {
      for (let v = 0; v < variantsPerDigit; v++) {
        dataset.push({ input: SeedDigits.renderSample(digit), label: digit });
      }
    }
    return dataset;
  },
};