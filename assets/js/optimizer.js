"use strict";

class Optimizer {
  constructor(type = "gradientDescent", learningRate = 0.1) {
    if (!isString(type)) throw new TypeError("Optimizer: type must be a string");
    if (!isPositive(learningRate)) throw new RangeError("Optimizer: learningRate must be positive");
    this.type = type;
    this.learningRate = learningRate;
  }

  /**
   * Applies one gradient descent step to a layer in place.
   * @param {Layer} layer
   * @param {Matrix} gradientWeights - dLoss/dWeights, same shape as layer.weights
   * @param {Matrix} gradientBiases - dLoss/dBiases, same shape as layer.biases
   */
  update(layer, gradientWeights, gradientBiases) {
    if (!(layer instanceof Layer)) throw new TypeError("Optimizer.update: layer must be a Layer instance");
    layer.weights = Matrix.subtract(layer.weights, Matrix.scalar(gradientWeights, this.learningRate));
    layer.biases = Matrix.subtract(layer.biases, Matrix.scalar(gradientBiases, this.learningRate));
  }
}