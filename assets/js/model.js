"use strict";

const ModelConfig = {
  inputSize: 784,
  hiddenLayers: [
    { size: 128, activation: "relu" },
    { size: 64, activation: "relu" },
  ],
  outputLayer: { size: 10, activation: "softmax" },
};

const ModelIO = {
  saveModel(network) {
    if (!(network instanceof NeuralNetwork)) {
      throw new TypeError("ModelIO.saveModel: network must be a NeuralNetwork instance");
    }
    const payload = {
      config: network.config,
      learningRate: network.optimizer.learningRate,
      trainingSteps: network.trainingSteps,
      layers: network.layers.map((layer) => ({
        weights: layer.weights.data,
        biases: layer.biases.data,
        activation: layer.activationName,
      })),
      savedAt: timestamp(),
    };
    return JSON.stringify(payload);
  },

  loadModel(json) {
    if (!isString(json)) throw new TypeError("ModelIO.loadModel: json must be a string");
    const payload = JSON.parse(json);
    const optimizer = new Optimizer("gradientDescent", payload.learningRate || 0.1);
    const network = new NeuralNetwork(payload.config, optimizer);
    payload.layers.forEach((layerData, i) => {
      network.layers[i].weights.data = layerData.weights;
      network.layers[i].biases.data = layerData.biases;
    });
    network.trainingSteps = payload.trainingSteps || 0;
    return network;
  },

  exportWeights(network) {
    if (!(network instanceof NeuralNetwork)) {
      throw new TypeError("ModelIO.exportWeights: network must be a NeuralNetwork instance");
    }
    return network.layers.map((layer) => ({
      weights: cloneMatrix(layer.weights.data),
      biases: cloneMatrix(layer.biases.data),
    }));
  },

  importWeights(network, weightsArray) {
    if (!(network instanceof NeuralNetwork)) {
      throw new TypeError("ModelIO.importWeights: network must be a NeuralNetwork instance");
    }
    if (!isArray(weightsArray) || weightsArray.length !== network.layers.length) {
      throw new RangeError("ModelIO.importWeights: weightsArray must match layer count");
    }
    weightsArray.forEach((w, i) => {
      network.layers[i].weights.data = w.weights;
      network.layers[i].biases.data = w.biases;
    });
  },
};