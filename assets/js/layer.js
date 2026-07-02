"use strict";

/**
 * Layer
 * A single dense (fully connected) layer. Owns its weights, biases,
 * and the last computed activations so the network can expose them
 * later for visualization or debugging.
 */
class Layer {
  constructor(inputSize, outputSize, activationName = "relu") {
    if (!Number.isInteger(inputSize) || !Number.isInteger(outputSize)) {
      throw new TypeError("Layer: inputSize and outputSize must be integers");
    }
    if (!Activations[activationName]) {
      throw new Error(`Layer: unknown activation "${activationName}"`);
    }

    this.inputSize = inputSize;
    this.outputSize = outputSize;
    this.activationName = activationName;

    this.weights = Matrix.heInit(outputSize, inputSize);
    this.biases = new Matrix(outputSize, 1);

    this.lastInput = null;
    this.lastZ = null;
    this.lastOutput = null;
  }

  forward(inputMatrix) {
    if (!(inputMatrix instanceof Matrix)) {
      throw new TypeError("Layer.forward: inputMatrix must be a Matrix instance");
    }

    this.lastInput = inputMatrix;

    let z = Matrix.multiply(this.weights, inputMatrix);
    z = Matrix.add(z, this.biases);
    this.lastZ = z;

    if (this.activationName === "softmax") {
      this.lastOutput = Matrix.fromArray(Activations.softmax.fn(z.toArray()));
    } else {
      this.lastOutput = Matrix.mapStatic(z, Activations[this.activationName].fn);
    }

    return this.lastOutput;
  }
}