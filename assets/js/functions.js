"use strict";

/* ==================================================
   NeuralGrid Engine — functions.js
   Central reusable utility library.
   No app logic, no UI, no NN classes — pure helpers
   consumed by matrix.js, layer.js, neuralNetwork.js,
   inputProcessor.js, renderer.js, and future modules.
   ================================================== */

/* ==================================================
   Math Utilities
   ==================================================
   Core numeric building blocks used by activation
   functions, normalization, and future training code
   (gradient descent, loss scaling, weight init). */

/**
 * Clamps a value between a minimum and maximum.
 * @param {number} value
 * @param {number} min
 * @param {number} max
 * @returns {number}
 * @example clamp(15, 0, 10) // 10
 */
const clamp = (value, min, max) => {
  if (typeof value !== "number" || typeof min !== "number" || typeof max !== "number") {
    throw new TypeError("clamp: value, min, and max must be numbers");
  }
  if (min > max) throw new RangeError("clamp: min cannot be greater than max");
  return Math.min(Math.max(value, min), max);
};

/**
 * Linearly interpolates between a and b by t.
 * @param {number} a - start value
 * @param {number} b - end value
 * @param {number} t - interpolation factor (typically 0..1)
 * @returns {number}
 * @example lerp(0, 10, 0.5) // 5
 */
const lerp = (a, b, t) => {
  if (![a, b, t].every((v) => typeof v === "number")) {
    throw new TypeError("lerp: a, b, and t must be numbers");
  }
  return a + (b - a) * t;
};

/**
 * Inverse of lerp: given a value between a and b, returns t.
 * @param {number} a
 * @param {number} b
 * @param {number} value
 * @returns {number}
 * @example inverseLerp(0, 10, 5) // 0.5
 */
const inverseLerp = (a, b, value) => {
  if (![a, b, value].every((v) => typeof v === "number")) {
    throw new TypeError("inverseLerp: a, b, and value must be numbers");
  }
  if (a === b) throw new RangeError("inverseLerp: a and b cannot be equal");
  return (value - a) / (b - a);
};

/**
 * Maps a value from one range to another.
 * @param {number} value
 * @param {number} inMin
 * @param {number} inMax
 * @param {number} outMin
 * @param {number} outMax
 * @returns {number}
 * @example mapRange(5, 0, 10, 0, 100) // 50
 */
const mapRange = (value, inMin, inMax, outMin, outMax) => {
  if (![value, inMin, inMax, outMin, outMax].every((v) => typeof v === "number")) {
    throw new TypeError("mapRange: all arguments must be numbers");
  }
  if (inMin === inMax) throw new RangeError("mapRange: inMin cannot equal inMax");
  const t = inverseLerp(inMin, inMax, value);
  return lerp(outMin, outMax, t);
};

/**
 * Normalizes a value from an arbitrary range into 0..1.
 * @param {number} value
 * @param {number} min
 * @param {number} max
 * @returns {number}
 * @example normalize(50, 0, 100) // 0.5
 */
const normalize = (value, min, max) => {
  if (![value, min, max].every((v) => typeof v === "number")) {
    throw new TypeError("normalize: value, min, and max must be numbers");
  }
  if (min === max) throw new RangeError("normalize: min cannot equal max");
  return (value - min) / (max - min);
};

/**
 * Alias-style remap, identical semantics to mapRange but kept
 * separate for API clarity when remapping already-normalized data.
 * @param {number} value
 * @param {number} inMin
 * @param {number} inMax
 * @param {number} outMin
 * @param {number} outMax
 * @returns {number}
 */
const remap = (value, inMin, inMax, outMin, outMax) =>
  mapRange(value, inMin, inMax, outMin, outMax);

/**
 * Sigmoid activation function.
 * @param {number} x
 * @returns {number} value in (0, 1)
 * @example sigmoid(0) // 0.5
 */
const sigmoid = (x) => {
  if (typeof x !== "number") throw new TypeError("sigmoid: x must be a number");
  return 1 / (1 + Math.exp(-x));
};

/**
 * Derivative of the sigmoid function, expressed in terms of x.
 * @param {number} x
 * @returns {number}
 */
const sigmoidDerivative = (x) => {
  if (typeof x !== "number") throw new TypeError("sigmoidDerivative: x must be a number");
  const s = sigmoid(x);
  return s * (1 - s);
};

/**
 * ReLU activation function.
 * @param {number} x
 * @returns {number}
 * @example relu(-5) // 0
 */
const relu = (x) => {
  if (typeof x !== "number") throw new TypeError("relu: x must be a number");
  return Math.max(0, x);
};

/**
 * Derivative of ReLU.
 * @param {number} x
 * @returns {number} 1 if x > 0, otherwise 0
 */
const reluDerivative = (x) => {
  if (typeof x !== "number") throw new TypeError("reluDerivative: x must be a number");
  return x > 0 ? 1 : 0;
};

/**
 * Hyperbolic tangent activation function.
 * @param {number} x
 * @returns {number} value in (-1, 1)
 */
const tanhActivation = (x) => {
  if (typeof x !== "number") throw new TypeError("tanhActivation: x must be a number");
  return Math.tanh(x);
};

/**
 * Derivative of tanh.
 * @param {number} x
 * @returns {number}
 */
const tanhDerivative = (x) => {
  if (typeof x !== "number") throw new TypeError("tanhDerivative: x must be a number");
  return 1 - Math.tanh(x) ** 2;
};

/**
 * Numerically stable softmax over an array of logits.
 * @param {number[]} values
 * @returns {number[]} probability distribution summing to 1
 * @example softmax([1, 2, 3]) // [0.09, 0.24, 0.67]
 */
const softmax = (values) => {
  if (!Array.isArray(values) || values.length === 0) {
    throw new TypeError("softmax: values must be a non-empty array");
  }
  if (!values.every((v) => typeof v === "number")) {
    throw new TypeError("softmax: all elements must be numbers");
  }
  const max = Math.max(...values);
  const exponentials = values.map((v) => Math.exp(v - max));
  const total = exponentials.reduce((acc, v) => acc + v, 0);
  return exponentials.map((v) => v / total);
};

/**
 * Returns a random floating point number in [min, max).
 * @param {number} min
 * @param {number} max
 * @returns {number}
 */
const randomFloat = (min = 0, max = 1) => {
  if (typeof min !== "number" || typeof max !== "number") {
    throw new TypeError("randomFloat: min and max must be numbers");
  }
  if (min > max) throw new RangeError("randomFloat: min cannot be greater than max");
  return Math.random() * (max - min) + min;
};

/**
 * Returns a random integer in [min, max] (inclusive).
 * @param {number} min
 * @param {number} max
 * @returns {number}
 */
const randomInt = (min, max) => {
  if (!Number.isInteger(min) || !Number.isInteger(max)) {
    throw new TypeError("randomInt: min and max must be integers");
  }
  if (min > max) throw new RangeError("randomInt: min cannot be greater than max");
  return Math.floor(Math.random() * (max - min + 1)) + min;
};

/**
 * Generates a random number from a Gaussian (normal) distribution
 * using the Box-Muller transform. Useful for weight initialization.
 * @param {number} [mean=0]
 * @param {number} [standardDeviation=1]
 * @returns {number}
 */
const gaussianRandom = (mean = 0, standardDeviation = 1) => {
  if (typeof mean !== "number" || typeof standardDeviation !== "number") {
    throw new TypeError("gaussianRandom: mean and standardDeviation must be numbers");
  }
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  const standardNormal = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  return mean + standardNormal * standardDeviation;
};

/**
 * Converts degrees to radians.
 * @param {number} degrees
 * @returns {number}
 */
const degreesToRadians = (degrees) => {
  if (typeof degrees !== "number") throw new TypeError("degreesToRadians: degrees must be a number");
  return (degrees * Math.PI) / 180;
};

/**
 * Converts radians to degrees.
 * @param {number} radians
 * @returns {number}
 */
const radiansToDegrees = (radians) => {
  if (typeof radians !== "number") throw new TypeError("radiansToDegrees: radians must be a number");
  return (radians * 180) / Math.PI;
};

/**
 * Checks whether two numbers are approximately equal within an epsilon.
 * @param {number} a
 * @param {number} b
 * @param {number} [epsilon=1e-9]
 * @returns {boolean}
 */
const nearlyEqual = (a, b, epsilon = 1e-9) => {
  if (typeof a !== "number" || typeof b !== "number") {
    throw new TypeError("nearlyEqual: a and b must be numbers");
  }
  return Math.abs(a - b) < epsilon;
};

/* ==================================================
   Array Utilities
   ==================================================
   Generic array operations used for dataset handling,
   batching, and shuffling training samples. */

/**
 * Sums all numeric elements of an array.
 * @param {number[]} array
 * @returns {number}
 */
const sum = (array) => {
  if (!Array.isArray(array)) throw new TypeError("sum: array must be an array");
  return array.reduce((acc, value) => {
    if (typeof value !== "number") throw new TypeError("sum: all elements must be numbers");
    return acc + value;
  }, 0);
};

/**
 * Computes the arithmetic mean of an array.
 * @param {number[]} array
 * @returns {number}
 */
const average = (array) => {
  if (!Array.isArray(array) || array.length === 0) {
    throw new TypeError("average: array must be a non-empty array");
  }
  return sum(array) / array.length;
};

/**
 * Returns the minimum value in an array.
 * @param {number[]} array
 * @returns {number}
 */
const min = (array) => {
  if (!Array.isArray(array) || array.length === 0) {
    throw new TypeError("min: array must be a non-empty array");
  }
  return array.reduce((acc, value) => (value < acc ? value : acc), array[0]);
};

/**
 * Returns the maximum value in an array.
 * @param {number[]} array
 * @returns {number}
 */
const max = (array) => {
  if (!Array.isArray(array) || array.length === 0) {
    throw new TypeError("max: array must be a non-empty array");
  }
  return array.reduce((acc, value) => (value > acc ? value : acc), array[0]);
};

/**
 * Recursively flattens a nested array to a single depth.
 * @param {Array} array
 * @param {number} [depth=Infinity]
 * @returns {Array}
 */
const flatten = (array, depth = Infinity) => {
  if (!Array.isArray(array)) throw new TypeError("flatten: array must be an array");
  return array.flat(depth);
};

/**
 * Splits an array into chunks of a given size.
 * @param {Array} array
 * @param {number} size
 * @returns {Array[]}
 * @example chunk([1,2,3,4,5], 2) // [[1,2],[3,4],[5]]
 */
const chunk = (array, size) => {
  if (!Array.isArray(array)) throw new TypeError("chunk: array must be an array");
  if (!Number.isInteger(size) || size <= 0) {
    throw new RangeError("chunk: size must be a positive integer");
  }
  const result = [];
  for (let i = 0; i < array.length; i += size) {
    result.push(array.slice(i, i + size));
  }
  return result;
};

/**
 * Returns an array of unique values, preserving first-seen order.
 * @param {Array} array
 * @returns {Array}
 */
const unique = (array) => {
  if (!Array.isArray(array)) throw new TypeError("unique: array must be an array");
  return [...new Set(array)];
};

/**
 * Returns a new shuffled copy of an array using Fisher-Yates.
 * @param {Array} array
 * @returns {Array}
 */
const shuffle = (array) => {
  if (!Array.isArray(array)) throw new TypeError("shuffle: array must be an array");
  const result = [...array];
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
};

/**
 * Returns a random sample of n elements from an array without replacement.
 * @param {Array} array
 * @param {number} count
 * @returns {Array}
 */
const sample = (array, count) => {
  if (!Array.isArray(array)) throw new TypeError("sample: array must be an array");
  if (!Number.isInteger(count) || count < 0) {
    throw new RangeError("sample: count must be a non-negative integer");
  }
  if (count > array.length) {
    throw new RangeError("sample: count cannot exceed array length");
  }
  return shuffle(array).slice(0, count);
};

/**
 * Zips multiple arrays together elementwise.
 * @param {...Array} arrays
 * @returns {Array[]}
 * @example zip([1,2],[3,4]) // [[1,3],[2,4]]
 */
const zip = (...arrays) => {
  if (arrays.length === 0) throw new TypeError("zip: at least one array is required");
  if (!arrays.every(Array.isArray)) throw new TypeError("zip: all arguments must be arrays");
  const length = Math.min(...arrays.map((a) => a.length));
  const result = [];
  for (let i = 0; i < length; i++) {
    result.push(arrays.map((a) => a[i]));
  }
  return result;
};

/**
 * Transposes a 2D array (rows become columns).
 * @param {Array[]} array2D
 * @returns {Array[]}
 */
const transposeArray = (array2D) => {
  if (!Array.isArray(array2D) || !array2D.every(Array.isArray)) {
    throw new TypeError("transposeArray: array2D must be a 2D array");
  }
  if (array2D.length === 0) return [];
  const rowLength = array2D[0].length;
  if (!array2D.every((row) => row.length === rowLength)) {
    throw new RangeError("transposeArray: all rows must have equal length");
  }
  return array2D[0].map((_, colIndex) => array2D.map((row) => row[colIndex]));
};

/**
 * Generates a numeric range array.
 * @param {number} start
 * @param {number} end
 * @param {number} [step=1]
 * @returns {number[]}
 * @example range(0, 5) // [0,1,2,3,4]
 */
const range = (start, end, step = 1) => {
  if (![start, end, step].every((v) => typeof v === "number")) {
    throw new TypeError("range: start, end, and step must be numbers");
  }
  if (step === 0) throw new RangeError("range: step cannot be zero");
  const result = [];
  if (step > 0) {
    for (let i = start; i < end; i += step) result.push(i);
  } else {
    for (let i = start; i > end; i += step) result.push(i);
  }
  return result;
};

/**
 * Creates an array of length n filled with a repeated value.
 * @param {*} value
 * @param {number} count
 * @returns {Array}
 */
const repeat = (value, count) => {
  if (!Number.isInteger(count) || count < 0) {
    throw new RangeError("repeat: count must be a non-negative integer");
  }
  return new Array(count).fill(value);
};

/**
 * Creates a deep copy of a (possibly nested) array via JSON round-trip.
 * Only safe for JSON-serializable data (numbers, strings, booleans, null).
 * @param {Array} array
 * @returns {Array}
 */
const deepCopyArray = (array) => {
  if (!Array.isArray(array)) throw new TypeError("deepCopyArray: array must be an array");
  return JSON.parse(JSON.stringify(array));
};

/* ==================================================
   Matrix Utilities
   ==================================================
   Lightweight helpers operating on plain 2D arrays.
   These support the future Matrix class and layer
   weight/bias initialization but do not replace it. */

/**
 * Creates a 2D array of given dimensions filled with a default value.
 * @param {number} rows
 * @param {number} cols
 * @param {number} [fillValue=0]
 * @returns {number[][]}
 */
const createMatrix = (rows, cols, fillValue = 0) => {
  if (!Number.isInteger(rows) || !Number.isInteger(cols) || rows <= 0 || cols <= 0) {
    throw new RangeError("createMatrix: rows and cols must be positive integers");
  }
  return Array.from({ length: rows }, () => new Array(cols).fill(fillValue));
};

/**
 * Deep clones a 2D array matrix.
 * @param {number[][]} matrix
 * @returns {number[][]}
 */
const cloneMatrix = (matrix) => {
  if (!isMatrix(matrix)) throw new TypeError("cloneMatrix: matrix must be a valid 2D array");
  return matrix.map((row) => [...row]);
};

/**
 * Creates an identity matrix of a given size.
 * @param {number} size
 * @returns {number[][]}
 */
const identityMatrix = (size) => {
  if (!Number.isInteger(size) || size <= 0) {
    throw new RangeError("identityMatrix: size must be a positive integer");
  }
  return createMatrix(size, size).map((row, i) => row.map((_, j) => (i === j ? 1 : 0)));
};

/**
 * Creates a matrix filled with random floats in [min, max).
 * @param {number} rows
 * @param {number} cols
 * @param {number} [min=-1]
 * @param {number} [max=1]
 * @returns {number[][]}
 */
const randomMatrix = (rows, cols, min = -1, max = 1) => {
  const matrix = createMatrix(rows, cols);
  return matrix.map((row) => row.map(() => randomFloat(min, max)));
};

/**
 * Creates a matrix filled entirely with zeros.
 * @param {number} rows
 * @param {number} cols
 * @returns {number[][]}
 */
const zerosMatrix = (rows, cols) => createMatrix(rows, cols, 0);

/**
 * Creates a matrix filled entirely with ones.
 * @param {number} rows
 * @param {number} cols
 * @returns {number[][]}
 */
const onesMatrix = (rows, cols) => createMatrix(rows, cols, 1);

/**
 * Returns the dimensions of a matrix as { rows, cols }.
 * @param {number[][]} matrix
 * @returns {{rows: number, cols: number}}
 */
const matrixDimensions = (matrix) => {
  if (!isMatrix(matrix)) throw new TypeError("matrixDimensions: matrix must be a valid 2D array");
  return { rows: matrix.length, cols: matrix[0].length };
};

/**
 * Checks whether a value is a valid rectangular 2D numeric array.
 * @param {*} value
 * @returns {boolean}
 */
const isMatrix = (value) => {
  if (!Array.isArray(value) || value.length === 0) return false;
  if (!value.every(Array.isArray)) return false;
  const rowLength = value[0].length;
  if (rowLength === 0) return false;
  return value.every(
    (row) => row.length === rowLength && row.every((cell) => typeof cell === "number")
  );
};

/**
 * Flattens a matrix into a 1D array (row-major order).
 * @param {number[][]} matrix
 * @returns {number[]}
 */
const flattenMatrix = (matrix) => {
  if (!isMatrix(matrix)) throw new TypeError("flattenMatrix: matrix must be a valid 2D array");
  return matrix.reduce((acc, row) => acc.concat(row), []);
};

/**
 * Reshapes a flat array into a 2D matrix of given dimensions.
 * @param {number[]} array
 * @param {number} rows
 * @param {number} cols
 * @returns {number[][]}
 */
const reshapeMatrix = (array, rows, cols) => {
  if (!Array.isArray(array)) throw new TypeError("reshapeMatrix: array must be an array");
  if (!Number.isInteger(rows) || !Number.isInteger(cols) || rows <= 0 || cols <= 0) {
    throw new RangeError("reshapeMatrix: rows and cols must be positive integers");
  }
  if (array.length !== rows * cols) {
    throw new RangeError("reshapeMatrix: array length must equal rows * cols");
  }
  const result = [];
  for (let i = 0; i < rows; i++) {
    result.push(array.slice(i * cols, i * cols + cols));
  }
  return result;
};

/**
 * Converts a single-column or single-row matrix to a flat array.
 * @param {number[][]} matrix
 * @returns {number[]}
 */
const matrixToArray = (matrix) => {
  if (!isMatrix(matrix)) throw new TypeError("matrixToArray: matrix must be a valid 2D array");
  const { rows, cols } = matrixDimensions(matrix);
  if (rows !== 1 && cols !== 1) {
    throw new RangeError("matrixToArray: matrix must be a single row or single column");
  }
  return flattenMatrix(matrix);
};

/**
 * Converts a flat array into a single-column matrix (Nx1).
 * @param {number[]} array
 * @returns {number[][]}
 */
const arrayToMatrix = (array) => {
  if (!Array.isArray(array) || array.length === 0) {
    throw new TypeError("arrayToMatrix: array must be a non-empty array");
  }
  return array.map((value) => [value]);
};

/* ==================================================
   Random Utilities
   ==================================================
   General-purpose randomness helpers used for IDs,
   sampling, augmentation, and stochastic training. */

/**
 * Generates a RFC4122-style UUID v4 string.
 * @returns {string}
 */
const randomUUID = () => {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (character) => {
    const random = (Math.random() * 16) | 0;
    const value = character === "x" ? random : (random & 0x3) | 0x8;
    return value.toString(16);
  });
};

/**
 * Returns a random element from an array.
 * @param {Array} array
 * @returns {*}
 */
const randomChoice = (array) => {
  if (!Array.isArray(array) || array.length === 0) {
    throw new TypeError("randomChoice: array must be a non-empty array");
  }
  return array[randomInt(0, array.length - 1)];
};

/**
 * Returns a random element from an array based on relative weights.
 * @param {Array} items
 * @param {number[]} weights
 * @returns {*}
 * @example weightedChoice(["a","b"], [1,3]) // "b" ~75% of the time
 */
const weightedChoice = (items, weights) => {
  if (!Array.isArray(items) || !Array.isArray(weights)) {
    throw new TypeError("weightedChoice: items and weights must be arrays");
  }
  if (items.length !== weights.length || items.length === 0) {
    throw new RangeError("weightedChoice: items and weights must be non-empty and equal length");
  }
  if (weights.some((w) => typeof w !== "number" || w < 0)) {
    throw new RangeError("weightedChoice: weights must be non-negative numbers");
  }
  const totalWeight = sum(weights);
  if (totalWeight === 0) throw new RangeError("weightedChoice: total weight cannot be zero");
  let threshold = Math.random() * totalWeight;
  for (let i = 0; i < items.length; i++) {
    threshold -= weights[i];
    if (threshold <= 0) return items[i];
  }
  return items[items.length - 1];
};

/**
 * Generates a random hex color string.
 * @returns {string} e.g. "#a3f2c1"
 */
const randomColor = () => {
  const hex = randomInt(0, 0xffffff).toString(16).padStart(6, "0");
  return `#${hex}`;
};

/**
 * Returns a random boolean value.
 * @param {number} [probabilityTrue=0.5]
 * @returns {boolean}
 */
const randomBoolean = (probabilityTrue = 0.5) => {
  if (typeof probabilityTrue !== "number" || probabilityTrue < 0 || probabilityTrue > 1) {
    throw new RangeError("randomBoolean: probabilityTrue must be between 0 and 1");
  }
  return Math.random() < probabilityTrue;
};

/**
 * Alias of gaussianRandom, exposed under the Random Utilities section
 * for API discoverability.
 * @param {number} [mean=0]
 * @param {number} [standardDeviation=1]
 * @returns {number}
 */
const randomGaussian = (mean = 0, standardDeviation = 1) => gaussianRandom(mean, standardDeviation);

/* ==================================================
   Validation Utilities
   ==================================================
   Type guards used throughout the engine to validate
   inputs before performing math or mutating state. */

/**
 * Checks whether a value is a finite number.
 * @param {*} value
 * @returns {boolean}
 */
const isNumber = (value) => typeof value === "number" && Number.isFinite(value);

/**
 * Checks whether a value is an integer.
 * @param {*} value
 * @returns {boolean}
 */
const isInteger = (value) => Number.isInteger(value);

/**
 * Checks whether a value is a non-integer finite number.
 * @param {*} value
 * @returns {boolean}
 */
const isFloat = (value) => isNumber(value) && !Number.isInteger(value);

/**
 * Checks whether a value is an array.
 * @param {*} value
 * @returns {boolean}
 */
const isArray = (value) => Array.isArray(value);

/**
 * Checks whether a value is a function.
 * @param {*} value
 * @returns {boolean}
 */
const isFunction = (value) => typeof value === "function";

/**
 * Checks whether a value is a string.
 * @param {*} value
 * @returns {boolean}
 */
const isString = (value) => typeof value === "string";

/**
 * Checks whether a value is a plain object (not array, not null).
 * @param {*} value
 * @returns {boolean}
 */
const isObject = (value) =>
  typeof value === "object" && value !== null && !Array.isArray(value);

/**
 * Checks whether a number is strictly positive.
 * @param {number} value
 * @returns {boolean}
 */
const isPositive = (value) => isNumber(value) && value > 0;

/**
 * Checks whether a number is strictly negative.
 * @param {number} value
 * @returns {boolean}
 */
const isNegative = (value) => isNumber(value) && value < 0;

/**
 * Checks whether a positive integer is a power of two.
 * @param {number} value
 * @returns {boolean}
 */
const isPowerOfTwo = (value) => {
  if (!Number.isInteger(value) || value <= 0) return false;
  return (value & (value - 1)) === 0;
};

/**
 * Checks whether a value is "empty" — covers arrays, strings,
 * objects, Map, Set, null, and undefined.
 * @param {*} value
 * @returns {boolean}
 */
const isEmpty = (value) => {
  if (value === null || value === undefined) return true;
  if (typeof value === "string" || Array.isArray(value)) return value.length === 0;
  if (value instanceof Map || value instanceof Set) return value.size === 0;
  if (isObject(value)) return Object.keys(value).length === 0;
  return false;
};

/* ==================================================
   Formatting Utilities
   ==================================================
   Presentation helpers used by future UI/debug panels
   to render numbers, timestamps, and file sizes. */

/**
 * Formats a number with thousands separators.
 * @param {number} value
 * @param {string} [locale="en-US"]
 * @returns {string}
 */
const formatNumber = (value, locale = "en-US") => {
  if (!isNumber(value)) throw new TypeError("formatNumber: value must be a finite number");
  return value.toLocaleString(locale);
};

/**
 * Rounds a number to a fixed number of decimal places.
 * @param {number} value
 * @param {number} [decimals=2]
 * @returns {number}
 */
const roundTo = (value, decimals = 2) => {
  if (!isNumber(value)) throw new TypeError("roundTo: value must be a finite number");
  if (!Number.isInteger(decimals) || decimals < 0) {
    throw new RangeError("roundTo: decimals must be a non-negative integer");
  }
  const factor = 10 ** decimals;
  return Math.round(value * factor) / factor;
};

/**
 * Formats a 0..1 fraction as a percentage string.
 * @param {number} value - fraction, e.g. 0.42
 * @param {number} [decimals=2]
 * @returns {string} e.g. "42.00%"
 */
const percentage = (value, decimals = 2) => {
  if (!isNumber(value)) throw new TypeError("percentage: value must be a finite number");
  return `${roundTo(value * 100, decimals).toFixed(decimals)}%`;
};

/**
 * Pads a number with leading zeros to a target length.
 * @param {number} value
 * @param {number} [length=2]
 * @returns {string}
 */
const padNumber = (value, length = 2) => {
  if (!isNumber(value)) throw new TypeError("padNumber: value must be a finite number");
  if (!Number.isInteger(length) || length < 0) {
    throw new RangeError("padNumber: length must be a non-negative integer");
  }
  return String(Math.trunc(value)).padStart(length, "0");
};

/**
 * Returns an ISO 8601 timestamp string for the current moment.
 * @returns {string}
 */
const timestamp = () => new Date().toISOString();

/**
 * Converts a byte count into a human-readable string.
 * @param {number} bytes
 * @param {number} [decimals=2]
 * @returns {string} e.g. "1.50 MB"
 */
const bytesToReadable = (bytes, decimals = 2) => {
  if (!isNumber(bytes) || bytes < 0) {
    throw new RangeError("bytesToReadable: bytes must be a non-negative number");
  }
  if (bytes === 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const exponent = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / 1024 ** exponent;
  return `${roundTo(value, decimals)} ${units[exponent]}`;
};

/* ==================================================
   Performance Utilities
   ==================================================
   Rate-limiting and caching helpers for expensive or
   frequently-invoked operations (e.g. render loops,
   resize handlers, repeated forward passes). */

/**
 * Creates a debounced version of a function that delays invocation
 * until after `wait` milliseconds have elapsed since the last call.
 * @param {Function} fn
 * @param {number} wait
 * @returns {Function}
 */
const debounce = (fn, wait) => {
  if (!isFunction(fn)) throw new TypeError("debounce: fn must be a function");
  if (!Number.isInteger(wait) || wait < 0) {
    throw new RangeError("debounce: wait must be a non-negative integer");
  }
  let timeoutId = null;
  return function debounced(...args) {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn.apply(this, args), wait);
  };
};

/**
 * Creates a throttled version of a function that invokes at most
 * once per `limit` milliseconds.
 * @param {Function} fn
 * @param {number} limit
 * @returns {Function}
 */
const throttle = (fn, limit) => {
  if (!isFunction(fn)) throw new TypeError("throttle: fn must be a function");
  if (!Number.isInteger(limit) || limit < 0) {
    throw new RangeError("throttle: limit must be a non-negative integer");
  }
  let isThrottled = false;
  let pendingArgs = null;
  let pendingContext = null;

  const invoke = (context, args) => {
    fn.apply(context, args);
    isThrottled = true;
    setTimeout(() => {
      isThrottled = false;
      if (pendingArgs) {
        invoke(pendingContext, pendingArgs);
        pendingArgs = null;
        pendingContext = null;
      }
    }, limit);
  };

  return function throttled(...args) {
    if (!isThrottled) {
      invoke(this, args);
    } else {
      pendingArgs = args;
      pendingContext = this;
    }
  };
};

/**
 * Creates a memoized version of a function using a JSON-serialized
 * argument list as the cache key. Best suited for pure functions
 * with JSON-serializable arguments.
 * @param {Function} fn
 * @returns {Function}
 */
const memoize = (fn) => {
  if (!isFunction(fn)) throw new TypeError("memoize: fn must be a function");
  const cache = new Map();
  return function memoized(...args) {
    const key = JSON.stringify(args);
    if (cache.has(key)) return cache.get(key);
    const result = fn.apply(this, args);
    cache.set(key, result);
    return result;
  };
};

/* ==================================================
   Debug Utilities
   ==================================================
   Development-time helpers for catching bugs early and
   measuring performance of engine internals. */

/**
 * Throws an error if the given condition is falsy.
 * @param {boolean} condition
 * @param {string} [message="Assertion failed"]
 * @returns {void}
 */
const assert = (condition, message = "Assertion failed") => {
  if (!condition) throw new Error(`assert: ${message}`);
};

/**
 * Recursively deep-freezes an object or array, preventing any
 * further mutation at any nesting level.
 * @param {*} value
 * @returns {*} the same value, now deeply frozen
 */
const deepFreeze = (value) => {
  if (value === null || (typeof value !== "object" && typeof value !== "function")) {
    return value;
  }
  Object.getOwnPropertyNames(value).forEach((property) => {
    const child = value[property];
    if (child !== null && (typeof child === "object" || typeof child === "function")) {
      deepFreeze(child);
    }
  });
  return Object.freeze(value);
};

/**
 * Measures the execution time of a function in milliseconds.
 * @param {Function} fn
 * @param {string} [label="execution"]
 * @returns {{result: *, durationMs: number}}
 */
const executionTimer = (fn, label = "execution") => {
  if (!isFunction(fn)) throw new TypeError("executionTimer: fn must be a function");
  const start = typeof performance !== "undefined" ? performance.now() : Date.now();
  const result = fn();
  const end = typeof performance !== "undefined" ? performance.now() : Date.now();
  const durationMs = end - start;
  // eslint-disable-next-line no-console
  console.log(`[executionTimer] ${label}: ${durationMs.toFixed(3)}ms`);
  return { result, durationMs };
};

/* ==================================================
   Neural Network Helpers
   ==================================================
   Generic, inference-free helpers shared by preprocessing,
   the network engine, and future training code. These do
   NOT perform forward propagation or any model logic. */

/**
 * Normalizes an array of raw pixel values (0..255) into 0..1.
 * @param {number[]} pixels
 * @returns {number[]}
 */
const normalizePixels = (pixels) => {
  if (!Array.isArray(pixels)) throw new TypeError("normalizePixels: pixels must be an array");
  return pixels.map((value) => {
    if (typeof value !== "number" || value < 0 || value > 255) {
      throw new RangeError("normalizePixels: each pixel must be a number in range 0..255");
    }
    return value / 255;
  });
};

/**
 * Flattens a 2D image (array of rows) into a 1D pixel array.
 * @param {number[][]} image
 * @returns {number[]}
 */
const flattenImage = (image) => {
  if (!Array.isArray(image) || !image.every(Array.isArray)) {
    throw new TypeError("flattenImage: image must be a 2D array");
  }
  return flattenMatrix(image);
};

/**
 * Returns the index of the maximum value in an array.
 * @param {number[]} array
 * @returns {number}
 */
const argMax = (array) => {
  if (!Array.isArray(array) || array.length === 0) {
    throw new TypeError("argMax: array must be a non-empty array");
  }
  let bestIndex = 0;
  for (let i = 1; i < array.length; i++) {
    if (array[i] > array[bestIndex]) bestIndex = i;
  }
  return bestIndex;
};

/**
 * Returns the index of the minimum value in an array.
 * @param {number[]} array
 * @returns {number}
 */
const argMin = (array) => {
  if (!Array.isArray(array) || array.length === 0) {
    throw new TypeError("argMin: array must be a non-empty array");
  }
  let bestIndex = 0;
  for (let i = 1; i < array.length; i++) {
    if (array[i] < array[bestIndex]) bestIndex = i;
  }
  return bestIndex;
};

/**
 * Returns the highest confidence value from a probability distribution.
 * @param {number[]} distribution
 * @returns {number}
 */
const confidenceScore = (distribution) => {
  if (!Array.isArray(distribution) || distribution.length === 0) {
    throw new TypeError("confidenceScore: distribution must be a non-empty array");
  }
  return max(distribution);
};

/**
 * Converts a class index into a one-hot encoded vector.
 * @param {number} index
 * @param {number} numClasses
 * @returns {number[]}
 * @example oneHotEncode(2, 5) // [0,0,1,0,0]
 */
const oneHotEncode = (index, numClasses) => {
  if (!Number.isInteger(index) || !Number.isInteger(numClasses)) {
    throw new TypeError("oneHotEncode: index and numClasses must be integers");
  }
  if (index < 0 || index >= numClasses) {
    throw new RangeError("oneHotEncode: index out of bounds for numClasses");
  }
  const vector = new Array(numClasses).fill(0);
  vector[index] = 1;
  return vector;
};

/**
 * Converts a one-hot encoded vector back into a class index.
 * @param {number[]} vector
 * @returns {number}
 */
const oneHotDecode = (vector) => {
  if (!Array.isArray(vector) || vector.length === 0) {
    throw new TypeError("oneHotDecode: vector must be a non-empty array");
  }
  return argMax(vector);
};

/* ==================================================
   Exports
   ================================================== */
if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    clamp, lerp, inverseLerp, mapRange, normalize, remap,
    sigmoid, sigmoidDerivative, relu, reluDerivative,
    tanhActivation, tanhDerivative, softmax,
    randomFloat, randomInt, gaussianRandom,
    degreesToRadians, radiansToDegrees, nearlyEqual,
    sum, average, min, max, flatten, chunk, unique, shuffle,
    sample, zip, transposeArray, range, repeat, deepCopyArray,
    createMatrix, cloneMatrix, identityMatrix, randomMatrix,
    zerosMatrix, onesMatrix, matrixDimensions, isMatrix,
    flattenMatrix, reshapeMatrix, matrixToArray, arrayToMatrix,
    randomUUID, randomChoice, weightedChoice, randomColor,
    randomBoolean, randomGaussian,
    isNumber, isInteger, isFloat, isArray, isFunction, isString,
    isObject, isPositive, isNegative, isPowerOfTwo, isEmpty,
    formatNumber, roundTo, percentage, padNumber, timestamp, bytesToReadable,
    debounce, throttle, memoize,
    assert, deepFreeze, executionTimer,
    normalizePixels, flattenImage, argMax, argMin,
    confidenceScore, oneHotEncode, oneHotDecode,
  };
}