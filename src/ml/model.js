import { MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH } from '../constants.js';
import { STATUS } from '../domRefs.js';
import { state } from '../state.js';

export async function loadMobileNetFeatureModel() {
  const URL =
    'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1';

  state.mobilenet = await tf.loadGraphModel(URL, { fromTFHub: true });
  if (STATUS) {
    STATUS.innerText = 'MobileNet v3 loaded successfully!';
  }

  tf.tidy(function () {
    const answer = state.mobilenet.predict(
      tf.zeros([1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3])
    );
    console.log(answer.shape);
  });
}

export function rebuildModel(inputSize) {
  const outputUnits = Math.max(state.classNames.length, 1);
  const lr = sanitizeLearningRate(state.trainingLearningRate);
  const featureSize =
    typeof inputSize === 'number' && inputSize > 0 ? inputSize : 1024;

  if (state.model) {
    state.model.dispose();
  }

  state.model = tf.sequential();
  state.model.add(tf.layers.dense({ inputShape: [featureSize], units: 128, activation: 'relu' }));
  state.model.add(tf.layers.dense({ units: outputUnits, activation: 'softmax' }));

  state.model.compile({
    optimizer: tf.train.adam(lr),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  state.model.summary();
}

function sanitizeLearningRate(value) {
  const lr = Number(value);
  if (!Number.isFinite(lr) || lr <= 0) return 0.001;
  return lr;
}
