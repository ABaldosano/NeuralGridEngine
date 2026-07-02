"use strict";

/**
 * Matrix
 * Core math primitive for layer computation (weights x inputs + biases).
 * Wraps the plain-array helpers in functions.js with object-oriented,
 * chainable operations used throughout the network engine.
 */
class Matrix {
  constructor(rows, cols) {
    if (!Number.isInteger(rows) || !Number.isInteger(cols) || rows <= 0 || cols <= 0) {
      throw new RangeError("Matrix: rows and cols must be positive integers");
    }
    this.rows = rows;
    this.cols = cols;
    this.data = createMatrix(rows, cols, 0);
  }

  static fromArray(array) {
    if (!Array.isArray(array) || array.length === 0) {
      throw new TypeError("Matrix.fromArray: array must be a non-empty array");
    }
    const matrix = new Matrix(array.length, 1);
    matrix.data = arrayToMatrix(array);
    return matrix;
  }

  toArray() {
    return flattenMatrix(this.data);
  }

  static randomize(rows, cols, min = -1, max = 1) {
    const matrix = new Matrix(rows, cols);
    matrix.data = randomMatrix(rows, cols, min, max);
    return matrix;
  }

  /**
   * He (Kaiming) initialization: weights drawn from N(0, sqrt(2/fanIn)).
   * Standard random init in [-1,1] is far too large for high-dimensional
   * inputs (e.g. 784-pixel images) — it saturates the first layer's
   * pre-activations, which saturates softmax at init and starves
   * backprop of usable gradient, causing training to collapse toward
   * a single output class regardless of input.
   * @param {number} rows - output size
   * @param {number} cols - input size (fan-in)
   * @returns {Matrix}
   */
  static heInit(rows, cols) {
    const matrix = new Matrix(rows, cols);
    const standardDeviation = Math.sqrt(2 / cols);
    matrix.data = matrix.data.map((row) => row.map(() => gaussianRandom(0, standardDeviation)));
    return matrix;
  }

  map(fn) {
    if (!isFunction(fn)) throw new TypeError("Matrix.map: fn must be a function");
    this.data = this.data.map((row, i) => row.map((value, j) => fn(value, i, j)));
    return this;
  }

  static mapStatic(matrixInstance, fn) {
    const result = new Matrix(matrixInstance.rows, matrixInstance.cols);
    result.data = matrixInstance.data.map((row, i) => row.map((value, j) => fn(value, i, j)));
    return result;
  }

  static multiply(a, b) {
    if (a.cols !== b.rows) {
      throw new Error(`Matrix.multiply: shape mismatch (${a.rows}x${a.cols}) * (${b.rows}x${b.cols})`);
    }
    const result = new Matrix(a.rows, b.cols);
    for (let i = 0; i < result.rows; i++) {
      for (let j = 0; j < result.cols; j++) {
        let total = 0;
        for (let k = 0; k < a.cols; k++) total += a.data[i][k] * b.data[k][j];
        result.data[i][j] = total;
      }
    }
    return result;
  }

  static add(a, b) {
    if (a.rows !== b.rows || a.cols !== b.cols) throw new Error("Matrix.add: shape mismatch");
    const result = new Matrix(a.rows, a.cols);
    result.data = a.data.map((row, i) => row.map((value, j) => value + b.data[i][j]));
    return result;
  }

  static subtract(a, b) {
    if (a.rows !== b.rows || a.cols !== b.cols) throw new Error("Matrix.subtract: shape mismatch");
    const result = new Matrix(a.rows, a.cols);
    result.data = a.data.map((row, i) => row.map((value, j) => value - b.data[i][j]));
    return result;
  }

  static transpose(matrixInstance) {
    const result = new Matrix(matrixInstance.cols, matrixInstance.rows);
    result.data = transposeArray(matrixInstance.data);
    return result;
  }

  static scalar(matrixInstance, scalarValue) {
    return Matrix.mapStatic(matrixInstance, (value) => value * scalarValue);
  }

  static hadamard(a, b) {
    if (a.rows !== b.rows || a.cols !== b.cols) throw new Error("Matrix.hadamard: shape mismatch");
    const result = new Matrix(a.rows, a.cols);
    result.data = a.data.map((row, i) => row.map((value, j) => value * b.data[i][j]));
    return result;
  }
}