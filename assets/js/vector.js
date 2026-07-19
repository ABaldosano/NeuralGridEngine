"use strict";

const Vector = {
  add: (a, b) => zip(a, b).map(([x, y]) => x + y),
  subtract: (a, b) => zip(a, b).map(([x, y]) => x - y),
  dot: (a, b) => sum(zip(a, b).map(([x, y]) => x * y)),
  scale: (a, scalar) => a.map((x) => x * scalar),
  magnitude: (a) => Math.sqrt(sum(a.map((x) => x * x))),
};