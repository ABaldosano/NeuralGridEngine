"use strict";

/**
 * Activations
 * Registry mapping activation names to forward/derivative pairs.
 * Element-wise activations reuse functions.js primitives; softmax
 * is exposed separately since it acts on a whole vector at once.
 */
const Activations = {
  relu: { fn: relu, dfn: reluDerivative },
  sigmoid: { fn: sigmoid, dfn: sigmoidDerivative },
  tanh: { fn: tanhActivation, dfn: tanhDerivative },
  softmax: { fn: softmax, dfn: null }, // vector-level, no per-element derivative here
};