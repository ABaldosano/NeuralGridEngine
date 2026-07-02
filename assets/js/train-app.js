"use strict";

/**
 * train-app.js — "Train Your Own Model" sandbox.
 * Fully isolated storage keys (LAB_*) so nothing here ever touches
 * the production recognizer's model or dataset.
 */
(function initTrainingLab() {
  const LAB_MODEL_KEY = "neuralgrid-lab-model";
  const LAB_DATASET_KEY = "neuralgrid-lab-dataset";

  const canvas = document.getElementById("drawCanvas");
  const clearBtn = document.getElementById("clearBtn");
  const digitGrid = document.getElementById("digitGrid");
  const datasetStats = document.getElementById("datasetStats");
  const clearDatasetBtn = document.getElementById("clearDatasetBtn");
  const epochsInput = document.getElementById("epochsInput");
  const lrInput = document.getElementById("lrInput");
  const trainBtn = document.getElementById("trainBtn");
  const trainingLog = document.getElementById("trainingLog");
  const exportBtn = document.getElementById("exportBtn");
  const importBtn = document.getElementById("importBtn");
  const importFileInput = document.getElementById("importFileInput");
  const modelNameInput = document.getElementById("modelNameInput");
  const resetModelBtn = document.getElementById("resetModelBtn");
  const testDrawCanvas = document.getElementById("testDrawCanvas");
  const testNetworkCanvas = document.getElementById("testNetworkCanvas");
  const testClearBtn = document.getElementById("testClearBtn");
  const testPredictBtn = document.getElementById("testPredictBtn");
  const testResult = document.getElementById("testResult");
  const testPredictionLabel = document.getElementById("testPredictionLabel");
  const testConfidenceLabel = document.getElementById("testConfidenceLabel");
  const testStatusMessage = document.getElementById("testStatusMessage");

  CanvasController.init(canvas);
  const testCanvasController = CanvasController.create(testDrawCanvas);
  NetworkViz.init(testNetworkCanvas);

  let dataset = loadDataset();
  let network = loadOrCreateNetwork();

  buildDigitGrid();
  renderDatasetStats();

  clearBtn.addEventListener("click", () => {
    CanvasController.clear();
    testResult.hidden = true;
    testStatusMessage.textContent = "";
  });

  clearDatasetBtn.addEventListener("click", () => {
    if (!confirm("Clear the entire lab dataset?")) return;
    dataset = [];
    persistDataset();
    renderDatasetStats();
    log("Dataset cleared.");
  });

  trainBtn.addEventListener("click", () => {
    if (dataset.length === 0) {
      log("Add at least one labeled sample before training.");
      return;
    }
    const epochs = clamp(parseInt(epochsInput.value, 10) || 60, 1, 1000);
    const learningRate = clamp(parseFloat(lrInput.value) || 0.05, 0.001, 1);

    log(`Training on ${dataset.length} samples for ${epochs} epochs (lr=${learningRate})…`);
    setTimeout(() => {
      network = new NeuralNetwork(ModelConfig, new Optimizer("gradientDescent", learningRate));
      const finalLoss = network.trainDataset(dataset, epochs);
      persistModel();
      log(`Done. Final epoch avg loss: ${roundTo(finalLoss, 4)}`);
    }, 10);
  });

  const LAB_MODEL_NAME_KEY = "neuralgrid-lab-model-name";
  if (modelNameInput) {
    modelNameInput.value = localStorage.getItem(LAB_MODEL_NAME_KEY) || "";
    modelNameInput.addEventListener("input", () => {
      localStorage.setItem(LAB_MODEL_NAME_KEY, modelNameInput.value);
    });
  }

  exportBtn.addEventListener("click", () => {
    const json = ModelIO.saveModel(network);
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const rawName = (modelNameInput && modelNameInput.value.trim()) || "neuralgrid-lab-model";
    const safeName = rawName.replace(/[^a-z0-9\-_]+/gi, "-").replace(/^-+|-+$/g, "") || "neuralgrid-lab-model";
    const a = document.createElement("a");
    a.href = url;
    a.download = `${safeName}.json`;
    a.click();
    URL.revokeObjectURL(url);
  });

  if (importBtn && importFileInput) {
    importBtn.addEventListener("click", () => importFileInput.click());

    importFileInput.addEventListener("change", () => {
      const file = importFileInput.files && importFileInput.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => {
        try {
          network = ModelIO.loadModel(reader.result);
          persistModel();
          const baseName = file.name.replace(/\.json$/i, "");
          if (modelNameInput) {
            modelNameInput.value = baseName;
            localStorage.setItem(LAB_MODEL_NAME_KEY, baseName);
          }
          log(`Imported model "${baseName}".`);
        } catch (err) {
          log(`Import failed: ${err.message || "invalid model file"}`);
        } finally {
          importFileInput.value = "";
        }
      };
      reader.onerror = () => log("Import failed: could not read file.");
      reader.readAsText(file);
    });
  }

  resetModelBtn.addEventListener("click", () => {
    if (!confirm("Reset the lab model? Dataset is kept.")) return;
    localStorage.removeItem(LAB_MODEL_KEY);
    network = new NeuralNetwork(ModelConfig);
    log("Lab model reset (untrained).");
  });

  /**
   * Predict button: same pipeline as the production recognizer's
   * app.js (canvas -> Preprocessing.canvasToInputVector -> network.predict),
   * except the network here is the lab model trained on the dataset
   * the user built above, not the baked-in production baseline.
   */
  testClearBtn.addEventListener("click", () => {
    testCanvasController.clear();
    testResult.hidden = true;
    testStatusMessage.textContent = "";
    NetworkViz.init(testNetworkCanvas);
  });

  testPredictBtn.addEventListener("click", () => {
    if (network.trainingSteps === 0) {
      testStatusMessage.textContent = "Train the model at least once before predicting.";
      testResult.hidden = true;
      return;
    }

    const inputVector = Preprocessing.canvasToInputVector(testDrawCanvas, 28);
    const result = network.predict(inputVector);

    testPredictionLabel.textContent = result.prediction;
    testConfidenceLabel.textContent = `Confidence: ${percentage(result.confidence)}`;
    testResult.hidden = false;
    testStatusMessage.textContent = "";
    NetworkViz.render(inputVector, network.getActivations(), result.prediction);
  });

  function buildDigitGrid() {
    for (let digit = 0; digit <= 9; digit++) {
      const button = document.createElement("button");
      button.textContent = digit;
      button.addEventListener("click", () => addSample(digit));
      digitGrid.appendChild(button);
    }
  }

  function addSample(label) {
    const inputVector = Preprocessing.canvasToInputVector(canvas, 28);
    dataset.push({ input: inputVector, label });
    persistDataset();
    renderDatasetStats();
    CanvasController.clear();
    log(`Added sample for digit ${label}. Dataset size: ${dataset.length}`);
  }

  function renderDatasetStats() {
    const counts = new Array(10).fill(0);
    dataset.forEach((sample) => counts[sample.label]++);
    datasetStats.innerHTML = "";
    counts.forEach((count, digit) => {
      const li = document.createElement("li");
      li.textContent = `Digit ${digit}: ${count} sample${count === 1 ? "" : "s"}`;
      datasetStats.appendChild(li);
    });
  }

  function log(message) {
    trainingLog.textContent += `[${timestamp()}] ${message}\n`;
    trainingLog.scrollTop = trainingLog.scrollHeight;
  }

  function persistModel() {
    try {
      localStorage.setItem(LAB_MODEL_KEY, ModelIO.saveModel(network));
    } catch (error) {
      console.error("Failed to persist lab model:", error);
    }
  }

  function persistDataset() {
    try {
      localStorage.setItem(LAB_DATASET_KEY, JSON.stringify(dataset));
    } catch (error) {
      console.error("Failed to persist lab dataset:", error);
    }
  }

  function loadDataset() {
    try {
      const saved = localStorage.getItem(LAB_DATASET_KEY);
      if (saved) return JSON.parse(saved);
    } catch (error) {
      console.error("Failed to load lab dataset, starting fresh:", error);
    }
    return [];
  }

  function loadOrCreateNetwork() {
    try {
      const saved = localStorage.getItem(LAB_MODEL_KEY);
      if (saved) return ModelIO.loadModel(saved);
    } catch (error) {
      console.error("Failed to load lab model, starting fresh:", error);
    }
    return new NeuralNetwork(ModelConfig);
  }
})();