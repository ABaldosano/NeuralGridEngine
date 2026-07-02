"use strict";

/**
 * Vector
 * Thin, semantically-named wrapper around functions.js array utilities
 * for operations that read more clearly as "vector math" in NN code.
 */
const Vector = {
  add: (a, b) => zip(a, b).map(([x, y]) => x + y),
  subtract: (a, b) => zip(a, b).map(([x, y]) => x - y),
  dot: (a, b) => sum(zip(a, b).map(([x, y]) => x * y)),
  scale: (a, scalar) => a.map((x) => x * scalar),
  magnitude: (a) => Math.sqrt(sum(a.map((x) => x * x))),
};