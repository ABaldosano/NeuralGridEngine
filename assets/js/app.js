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

  const canvas = document.getElementById("drawCanvas");
  const predictBtn = document.getElementById("predictBtn");
  const clearBtn = document.getElementById("clearBtn");
  const resultCard = document.getElementById("result");
  const predictionLabel = document.getElementById("predictionLabel");
  const confidenceLabel = document.getElementById("confidenceLabel");
  const statusMessage = document.getElementById("statusMessage");

  CanvasController.init(canvas);

  const network = loadNetwork();

  predictBtn.addEventListener("click", () => {
    const inputVector = Preprocessing.canvasToInputVector(canvas, 28);
    const result = network.predict(inputVector);

    predictionLabel.textContent = result.prediction;
    confidenceLabel.textContent = `Confidence: ${percentage(result.confidence)}`;
    resultCard.hidden = false;
    statusMessage.textContent = "";
  });

  clearBtn.addEventListener("click", () => {
    CanvasController.clear();
    resultCard.hidden = true;
    statusMessage.textContent = "";
  });

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