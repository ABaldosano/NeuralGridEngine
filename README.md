# NeuralGrid Engine

A handwritten digit recognizer built entirely from scratch in vanilla JavaScript. No TensorFlow.js, no PyTorch, no external ML libraries. The matrix math, backpropagation, activation functions, and the neural network itself are all hand-written, running directly in the browser.

## What it does

Draw a digit from 0 to 9 on the canvas and the network predicts what you drew automatically, about one hundred milisecond after you stop drawing, with a live visualization showing every neuron firing as the prediction happens. The production recognizer ships with Mercury 0.1, a model trained from real user samples rather than a static benchmark dataset.

There is also a full training lab where you can build your own dataset by hand, tune hyperparameters, train a network from a blank slate, test it, and export or import the resulting model as JSON.

## Features

- Handwriting recognition with a live, animated network activation viewer (input layer through hidden layers to output)
- Predictions run automatically one hundred milisecond after you finish drawing — no Predict button, just draw and read the result
- A from-scratch neural network: matrix operations, ReLU and softmax activations, gradient descent, backpropagation, all written manually
- An isolated "Train Your Own Model" sandbox that never touches the production model
- Model export and import as JSON, so a trained model can be swapped in as the new baseline
- Light and dark themes (light by default), with the preference saved locally
- Responsive layout tuned for phones as well as desktop, with a reordered mobile flow (draw → result → network activity) and a performance-optimized visualization so it stays smooth on mobile hardware

## Tech stack

Plain HTML, CSS, and JavaScript. No frameworks, no build step, no dependencies. Everything runs client-side in the browser, including training.

## Project structure

```
NeuralGridEngine/
├── index.html              # Production recognizer page
├── train.html               # Train Your Own Model sandbox
├── assets/
│   ├── css/
│   │   └── global.css       # All styling, light/dark theme variables
│   └── js/
│       ├── matrix.js         # Matrix operations
│       ├── vector.js         # Vector operations
│       ├── activation.js     # ReLU, softmax, and other activations
│       ├── loss.js           # Loss functions
│       ├── optimizer.js      # Gradient descent optimizer
│       ├── layer.js          # Network layer logic
│       ├── neuralNetwork.js  # Core network: forward pass, backprop, training
│       ├── model.js          # Model config
│       ├── preprocessing.js  # Canvas-to-input-vector conversion
│       ├── renderer.js       # Canvas drawing controller
│       ├── networkViz.js     # Live network activation visualization
│       ├── pretrainedModel.js # Baked-in Mercury 0.1 weights
│       ├── app.js            # Production recognizer logic
│       ├── train-app.js      # Training lab logic
│       ├── seedDigits.js     # Seed dataset for the training lab
│       └── theme.js          # Theme toggle and mobile nav
└── README.md
```

## Running it locally

There is no build step. Clone the repo and open `index.html` in a browser, or serve the folder with any static file server:

```bash
git clone https://github.com/ABaldosano/NeuralGridEngine.git
cd NeuralGridEngine
python3 -m http.server 8000
```

Then visit `http://localhost:8000`.

## Training your own model

Head to `train.html`. The workflow is:

1. Draw a digit and label it by clicking the matching number, building up a dataset
2. Set epochs and learning rate
3. Train the model
4. Test it on new drawings — the prediction runs automatically one hundred milisecond after you stop drawing, with the same live network visualization used on the recognizer page
5. Export the trained model as JSON, or import a previously saved one to keep training it

A model trained here can be converted into a new production baseline by swapping it into `pretrainedModel.js`.

## Why build this from scratch

Most digit recognizer demos lean on TensorFlow.js and treat the network as a black box. The point of this project was the opposite: write the matrix math, the forward pass, and backpropagation by hand, so the whole pipeline from a mouse stroke on a canvas to a predicted digit is visible and understandable, not hidden behind a library.

## Author

Built by Arthur Baldosano Jr.

- GitHub: [github.com/ABaldosano](https://github.com/ABaldosano)
- LinkedIn: [Arthur Baldosano Jr.](https://www.linkedin.com/in/arthur-v-baldosano-jr-2b5607406)
- Portfolio: [www.arthurr.gt.tc](https://www.arthurr.gt.tc)

## License

MIT
