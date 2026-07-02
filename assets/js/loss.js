"use strict";

/**
 * Loss
 * Loss functions for future training support. fn computes the scalar
 * loss; dfn computes the gradient with respect to predictions.
 */
const Loss = {
  crossEntropy: {
    fn: (predicted, target) =>
      -sum(zip(predicted, target).map(([p, t]) => t * Math.log(p + 1e-15))),
    dfn: (predicted, target) => zip(predicted, target).map(([p, t]) => p - t),
  },
  meanSquaredError: {
    fn: (predicted, target) =>
      average(zip(predicted, target).map(([p, t]) => (p - t) ** 2)),
    dfn: (predicted, target) =>
      zip(predicted, target).map(([p, t]) => (2 * (p - t)) / predicted.length),
  },
};