"use strict";

class NeuralNetwork {
  constructor(config, optimizer = new Optimizer("gradientDescent", 0.05)) {
    if (!isObject(config)) throw new TypeError("NeuralNetwork: config must be an object");
    if (!isInteger(config.inputSize) || config.inputSize <= 0) {
      throw new RangeError("NeuralNetwork: config.inputSize must be a positive integer");
    }
    if (!isArray(config.hiddenLayers)) {
      throw new TypeError("NeuralNetwork: config.hiddenLayers must be an array");
    }
    if (!isObject(config.outputLayer)) {
      throw new TypeError("NeuralNetwork: config.outputLayer must be an object");
    }
    if (!(optimizer instanceof Optimizer)) {
      throw new TypeError("NeuralNetwork: optimizer must be an Optimizer instance");
    }

    this.config = config;
    this.optimizer = optimizer;
    this.layers = [];
    this.trainingSteps = 0;

    let previousSize = config.inputSize;

    for (const hidden of config.hiddenLayers) {
      this.layers.push(new Layer(previousSize, hidden.size, hidden.activation || "relu"));
      previousSize = hidden.size;
    }

    this.layers.push(
      new Layer(previousSize, config.outputLayer.size, config.outputLayer.activation || "softmax")
    );
  }

  forward(inputArray) {
    if (!isArray(inputArray) || inputArray.length !== this.config.inputSize) {
      throw new RangeError(`NeuralNetwork.forward: expected input of length ${this.config.inputSize}`);
    }
    let current = Matrix.fromArray(inputArray);
    for (const layer of this.layers) current = layer.forward(current);
    return current.toArray();
  }

  predict(inputArray) {
    const output = this.forward(inputArray);
    const predictedIndex = argMax(output);
    return {
      prediction: predictedIndex,
      confidence: confidenceScore(output),
      distribution: output,
    };
  }

  /**
   * Performs one supervised training step (backpropagation + gradient
   * descent) on a single labeled example. Assumes a softmax output
   * layer paired with cross-entropy loss, so the output-layer delta
   * simplifies to (predicted - target).
   * @param {number[]} inputArray - normalized input vector
   * @param {number} targetIndex - correct class index (0-based)
   * @returns {number} the cross-entropy loss for this example, pre-update
   */
  train(inputArray, targetIndex) {
    if (!isInteger(targetIndex) || targetIndex < 0 || targetIndex >= this.config.outputLayer.size) {
      throw new RangeError(`NeuralNetwork.train: targetIndex must be in range 0..${this.config.outputLayer.size - 1}`);
    }

    const target = oneHotEncode(targetIndex, this.config.outputLayer.size);
    const output = this.forward(inputArray); // populates lastInput/lastZ/lastOutput on every layer
    const loss = Loss.crossEntropy.fn(output, target);

    // Output layer delta for softmax + cross-entropy: (predicted - target)
    let delta = Matrix.fromArray(Vector.subtract(output, target));

    for (let i = this.layers.length - 1; i >= 0; i--) {
      const layer = this.layers[i];

      const gradientWeights = Matrix.multiply(delta, Matrix.transpose(layer.lastInput));
      const gradientBiases = delta;

      // Propagate delta to the previous layer before this layer's
      // weights are overwritten by the optimizer.
      let propagatedDelta = null;
      if (i > 0) {
        const previousLayer = this.layers[i - 1];
        const weightsTransposed = Matrix.transpose(layer.weights);
        const rawPropagated = Matrix.multiply(weightsTransposed, delta);
        const activationDerivative = Activations[previousLayer.activationName].dfn;
        if (!activationDerivative) {
          throw new Error(`NeuralNetwork.train: activation "${previousLayer.activationName}" has no derivative`);
        }
        const derivativeAtZ = Matrix.mapStatic(previousLayer.lastZ, activationDerivative);
        propagatedDelta = Matrix.hadamard(rawPropagated, derivativeAtZ);
      }

      this.optimizer.update(layer, gradientWeights, gradientBiases);

      if (propagatedDelta) delta = propagatedDelta;
    }

    this.trainingSteps += 1;
    return loss;
  }

  /**
   * Trains on an entire labeled dataset for several shuffled epochs.
   * This exists specifically to avoid catastrophic forgetting: a
   * single train() call only nudges the network toward one example,
   * so with a handful of examples the most recent label dominates.
   * Looping over every stored example each epoch keeps all learned
   * classes represented in every round of updates.
   * @param {Array<{input:number[], label:number}>} dataset
   * @param {number} [epochs=40]
   * @returns {number} average loss over the final epoch
   */
  trainDataset(dataset, epochs = 40) {
    if (!isArray(dataset) || dataset.length === 0) {
      throw new TypeError("NeuralNetwork.trainDataset: dataset must be a non-empty array");
    }
    if (!isInteger(epochs) || epochs <= 0) {
      throw new RangeError("NeuralNetwork.trainDataset: epochs must be a positive integer");
    }

    let averageLoss = 0;

    for (let epoch = 0; epoch < epochs; epoch++) {
      const shuffled = shuffle(dataset);
      const epochLosses = shuffled.map((sample) => this.train(sample.input, sample.label));
      averageLoss = average(epochLosses);
    }

    return averageLoss;
  }

  getActivations() {
    return this.layers.map((layer) => ({
      inputSize: layer.inputSize,
      outputSize: layer.outputSize,
      activation: layer.activationName,
      output: layer.lastOutput ? layer.lastOutput.toArray() : null,
    }));
  }
}