"use strict";

/**
 * app.js
 * Wires the UI to the prediction + supervised-correction pipeline:
 *
 *   Canvas -> InputProcessor -> NeuralNetwork.predict()
 *          -> user confirms or corrects -> example stored in dataset
 *          -> network retrained from scratch on the FULL dataset
 *
 * Why retrain on the full dataset instead of just the new example:
 * a single train() call only nudges weights toward one label, so with
 * a handful of examples the most recently corrected digit dominates
 * and the network "forgets" everything learned before it. Keeping
 * every corrected example and re-running several shuffled epochs over
 * all of them each time keeps every learned digit represented in
 * every round of updates.
 */
(function initApp() {
  const MODEL_STORAGE_KEY = "neuralgrid-model";
  const DATASET_STORAGE_KEY = "neuralgrid-dataset";
  const SEEDED_FLAG_KEY = "neuralgrid-seeded";
  const EPOCHS_PER_RETRAIN = 60;
  const SEED_VARIANTS_PER_DIGIT = 25;
  const SEED_EPOCHS = 80;

  const canvas = document.getElementById("drawCanvas");
  const predictBtn = document.getElementById("predictBtn");
  const clearBtn = document.getElementById("clearBtn");
  const predictionLabel = document.getElementById("predictionLabel");
  const confidenceLabel = document.getElementById("confidenceLabel");
  const correctionPanel = document.getElementById("correction");
  const correctBtn = document.getElementById("correctBtn");
  const digitGrid = document.getElementById("digitGrid");
  const statusMessage = document.getElementById("statusMessage");
  const trainingCount = document.getElementById("trainingCount");
  const resetBtn = document.getElementById("resetBtn");

  CanvasController.init(canvas);

  let dataset = loadDataset();
  let network = loadOrCreateNetwork();
  let lastInputVector = null;
  let lastPrediction = null;

  buildDigitGrid();

  if (!localStorage.getItem(SEEDED_FLAG_KEY) && !localStorage.getItem(MODEL_STORAGE_KEY)) {
    statusMessage.textContent = "Learning digits 0-9 for the first time...";
    setTimeout(() => {
      const seedSet = SeedDigits.generateDataset(SEED_VARIANTS_PER_DIGIT);
      dataset = dataset.concat(seedSet);
      persistDataset();

      network = new NeuralNetwork(ModelConfig);
      const finalLoss = network.trainDataset(dataset, SEED_EPOCHS);
      persistNetwork();
      localStorage.setItem(SEEDED_FLAG_KEY, "1");
      updateTrainingCount();

      statusMessage.textContent = `Baseline knowledge of digits 0-9 learned (avg loss ${roundTo(finalLoss, 4)}). Draw a digit to try it out.`;
    }, 10);
  } else {
    updateTrainingCount();
  }

  predictBtn.addEventListener("click", () => {
    lastInputVector = InputProcessor.canvasToInputVector(canvas, 28);
    const result = network.predict(lastInputVector);
    lastPrediction = result.prediction;

    predictionLabel.textContent = result.prediction;
    confidenceLabel.textContent = percentage(result.confidence);

    statusMessage.textContent = "";
    correctionPanel.hidden = false;
  });

  clearBtn.addEventListener("click", () => {
    CanvasController.clear();
    predictionLabel.textContent = "-";
    confidenceLabel.textContent = "-";
    correctionPanel.hidden = true;
    statusMessage.textContent = "";
    lastInputVector = null;
    lastPrediction = null;
  });

  correctBtn.addEventListener("click", () => {
    trainOnLabel(lastPrediction);
  });

  resetBtn.addEventListener("click", () => {
    if (!confirm("This discards everything the network has learned. Continue?")) return;
    localStorage.removeItem(MODEL_STORAGE_KEY);
    localStorage.removeItem(DATASET_STORAGE_KEY);
    localStorage.removeItem(SEEDED_FLAG_KEY);
    dataset = [];

    statusMessage.textContent = "Re-learning digits 0-9...";
    setTimeout(() => {
      const seedSet = SeedDigits.generateDataset(SEED_VARIANTS_PER_DIGIT);
      dataset = dataset.concat(seedSet);
      persistDataset();

      network = new NeuralNetwork(ModelConfig);
      const finalLoss = network.trainDataset(dataset, SEED_EPOCHS);
      persistNetwork();
      localStorage.setItem(SEEDED_FLAG_KEY, "1");
      updateTrainingCount();

      statusMessage.textContent = `Model reset. Baseline knowledge of digits 0-9 learned (avg loss ${roundTo(finalLoss, 4)}).`;
    }, 10);
  });

  /**
   * Builds the 0-9 digit correction buttons once.
   */
  function buildDigitGrid() {
    for (let digit = 0; digit <= 9; digit++) {
      const button = document.createElement("button");
      button.textContent = digit;
      button.addEventListener("click", () => trainOnLabel(digit));
      digitGrid.appendChild(button);
    }
  }

  /**
   * Records the current drawing with its true label, then retrains
   * the network from scratch over the ENTIRE accumulated dataset for
   * several shuffled epochs. This is what prevents the network from
   * collapsing onto whichever digit was corrected most recently.
   * @param {number} trueLabel
   */
  function trainOnLabel(trueLabel) {
    if (!lastInputVector) return;

    dataset.push({ input: lastInputVector, label: trueLabel });
    persistDataset();

    statusMessage.textContent = "Retraining on all examples...";
    correctionPanel.hidden = true;

    // Defer so the "Retraining..." message actually paints before the
    // (synchronous, blocking) retrain loop runs.
    setTimeout(() => {
      network = new NeuralNetwork(ModelConfig);
      const finalLoss = network.trainDataset(dataset, EPOCHS_PER_RETRAIN);
      persistNetwork();
      updateTrainingCount();

      statusMessage.textContent =
        `Retrained on ${dataset.length} example${dataset.length === 1 ? "" : "s"} ` +
        `(${EPOCHS_PER_RETRAIN} epochs, avg loss ${roundTo(finalLoss, 4)}).`;
    }, 10);
  }

  function updateTrainingCount() {
    trainingCount.textContent = `${dataset.length} training example${dataset.length === 1 ? "" : "s"} stored`;
  }

  function persistNetwork() {
    try {
      localStorage.setItem(MODEL_STORAGE_KEY, ModelIO.saveModel(network));
    } catch (error) {
      console.error("Failed to persist model:", error);
    }
  }

  function persistDataset() {
    try {
      localStorage.setItem(DATASET_STORAGE_KEY, JSON.stringify(dataset));
    } catch (error) {
      console.error("Failed to persist training dataset:", error);
    }
  }

  function loadDataset() {
    try {
      const saved = localStorage.getItem(DATASET_STORAGE_KEY);
      if (saved) return JSON.parse(saved);
    } catch (error) {
      console.error("Failed to load saved dataset, starting fresh:", error);
    }
    return [];
  }

  function loadOrCreateNetwork() {
    try {
      const saved = localStorage.getItem(MODEL_STORAGE_KEY);
      if (saved) return ModelIO.loadModel(saved);
    } catch (error) {
      console.error("Failed to load saved model, starting fresh:", error);
    }
    return new NeuralNetwork(ModelConfig);
  }
})();