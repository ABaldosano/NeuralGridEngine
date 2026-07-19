"use strict";

/**
 * app.js — Production recognizer only.
 * No training controls are exposed here. The shipped model is baked
 * offline (see pretrainedModel.js) and loaded verbatim — there is no
 * in-browser training on load, so the recognizer is ready instantly
 * and the page never freezes or shows a loading wait.
 */
(function initRecognizerApp() {
  const RECOGNIZER_MODEL_KEY = "neuralgrid-recognizer-model";
  const AUTO_PREDICT_DELAY_MS = 100;

  const canvas = document.getElementById("drawCanvas");
  const clearBtn = document.getElementById("clearBtn");
  const resultCard = document.getElementById("result");
  const predictionLabel = document.getElementById("predictionLabel");
  const confidenceLabel = document.getElementById("confidenceLabel");
  const statusMessage = document.getElementById("statusMessage");

  CanvasController.init(canvas);
  NetworkViz.init(document.getElementById("networkCanvas"));

  const network = loadNetwork();
  let autoPredictTimer = null;

  /**
   * There is no Predict button: a prediction runs automatically once
   * the user pauses for a second after a stroke. Every new stroke
   * restarts the timer, so mid-drawing pauses (lifting the pen between
   * strokes of the same digit) don't trigger a premature read — only
   * a full second of inactivity after the *last* stroke does.
   */
  CanvasController.onStrokeEnd = () => {
    if (autoPredictTimer) clearTimeout(autoPredictTimer);
    statusMessage.textContent = "Reading…";
    autoPredictTimer = setTimeout(runPrediction, AUTO_PREDICT_DELAY_MS);
  };

  clearBtn.addEventListener("click", () => {
    if (autoPredictTimer) {
      clearTimeout(autoPredictTimer);
      autoPredictTimer = null;
    }
    CanvasController.clear();
    predictionLabel.textContent = "-";
    confidenceLabel.textContent = "";
    statusMessage.textContent = "";
    resultCard.hidden = true;
    NetworkViz.init(document.getElementById("networkCanvas"));
  });

  function runPrediction() {
    autoPredictTimer = null;
    const inputVector = Preprocessing.canvasToInputVector(canvas, 28);
    const result = network.predict(inputVector);

    predictionLabel.textContent = result.prediction;
    confidenceLabel.textContent = `Confidence: ${percentage(result.confidence)}`;
    statusMessage.textContent = "";
    resultCard.hidden = false;
    NetworkViz.render(inputVector, network.getActivations(), result.prediction);
  }

  /**
   * Loads the user's locally-corrected model if one exists (from a
   * prior visit), otherwise loads the baked-in pretrained baseline.
   * Either way this is synchronous JSON parsing — no training here.
   */
  function loadNetwork() {
    try {
      const saved = localStorage.getItem(RECOGNIZER_MODEL_KEY);
      if (saved) return ModelIO.loadModel(saved);
    } catch (error) {
      console.error("Failed to load saved recognizer model, using baked baseline:", error);
    }
    return ModelIO.loadModel(PRETRAINED_MODEL_JSON);
  }
})();