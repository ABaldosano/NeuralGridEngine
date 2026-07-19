"use strict";

const Activations = {
  relu: { fn: relu, dfn: reluDerivative },
  sigmoid: { fn: sigmoid, dfn: sigmoidDerivative },
  tanh: { fn: tanhActivation, dfn: tanhDerivative },
  softmax: { fn: softmax, dfn: null }, // vector-level, no per-element derivative here
};