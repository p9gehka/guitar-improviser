import { MetalString } from './helpers-link/guitar/key3.5.js';
import { autoCorrelate, noteFromPitch } from './helpers-link/auto-correlate.js';
const CHECKPOINT_URL = 'https://storage.googleapis.com/download.magenta.tensorflow.org/models/performance_rnn/tfjs';

let lstmKernel1;
let lstmBias1;
let lstmKernel2;
let lstmBias2;
let lstmKernel3;
let lstmBias3;
let fcB;
let fcW;
let c, h;
const forgetBias = tf.scalar(1.0);

const MIN_MIDI_PITCH = 0;
const MAX_MIDI_PITCH = 127;
const MAX_SHIFT_STEPS = 100;
const VELOCITY_BINS = 32;

let noteDensityEncoding;
let pitchHistogramEncoding;

const EVENT_RANGES = [
  ['note_on', MIN_MIDI_PITCH, MAX_MIDI_PITCH],
  ['note_off', MIN_MIDI_PITCH, MAX_MIDI_PITCH],
  ['time_shift', 1, MAX_SHIFT_STEPS],
  ['velocity_change', 1, VELOCITY_BINS],
];

function calculateEventSize() {
  let eventOffset = 0;
  for (const eventRange of EVENT_RANGES) {
    const minValue = eventRange[1];
    const maxValue = eventRange[2];
    eventOffset += maxValue - minValue + 1;
  }
  return eventOffset;
}

const EVENT_SIZE = calculateEventSize();

const PRIMER_IDX = 355;  // shift 1s.

// How many steps to generate per generateStep call.
// Generating more steps makes it less likely that we'll lag behind in note
// generation. Generating fewer steps makes it less likely that the browser UI
// thread will be starved for cycles.
const STEPS_PER_GENERATE_CALL = 10;

let lastSample = tf.scalar(PRIMER_IDX, 'int32');

let guitar;
let ctx;
let source;
let arrayData;
let analyser;
let audioBuffer;
async function start() {
    ctx = new AudioContext({ sampleRate: 44100 });
    guitar = [new MetalString(ctx, -0.5), new MetalString(ctx, -0.5)];
    const comperessorNode = ctx.createDynamicsCompressor();
    const gainNode = ctx.createGain();
    gainNode.gain.setValueAtTime(0.05, ctx.currentTime)
    await Promise.all([...guitar].map(metalString => metalString.connect(comperessorNode)));

    comperessorNode.connect(gainNode);
    gainNode.connect(ctx.destination);

    analyser = ctx.createAnalyser();
    analyser.fftSize = 2048;
    source = ctx.createBufferSource();

    arrayData = new Float32Array(2048);
    source.connect(analyser);
    analyser.connect(ctx.destination)

    const response = await fetch(`${CHECKPOINT_URL}/weights_manifest.json`)
    const manifest = await response.json();
    const vars = await tf.loadWeights(manifest, CHECKPOINT_URL);

    lstmKernel1 = vars['rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel'];
    lstmBias1 = vars['rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias'];
    lstmKernel2 = vars['rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel'];
    lstmBias2 = vars['rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias'];
    lstmKernel3 = vars['rnn/multi_rnn_cell/cell_2/basic_lstm_cell/kernel'];
    lstmBias3 = vars['rnn/multi_rnn_cell/cell_2/basic_lstm_cell/bias'];

    fcB = vars['fully_connected/biases'];
    fcW = vars['fully_connected/weights'];
    ///modelReady = true;
    resetRnn();
    console.log('Loaded')
}

function resetRnn() {
  c = [
    tf.zeros([1, lstmBias1.shape[0] / 4]),
    tf.zeros([1, lstmBias2.shape[0] / 4]),
    tf.zeros([1, lstmBias3.shape[0] / 4]),
  ];
  h = [
    tf.zeros([1, lstmBias1.shape[0] / 4]),
    tf.zeros([1, lstmBias2.shape[0] / 4]),
    tf.zeros([1, lstmBias3.shape[0] / 4]),
  ];

  lastSample?.dispose();
  lastSample = tf.scalar(PRIMER_IDX, 'int32');
}

function getConditioning() {
  return tf.tidy(() => {
      const axis = 0;
      const conditioningValues = noteDensityEncoding.concat(pitchHistogramEncoding, axis);
      return tf.tensor1d([0], 'int32').concat(conditioningValues, axis);
  });
}

function generate() {
    playOutPos = 0;
    serias = new Float32Array(partSize).map((v) => NaN);
    while (playOutPos < partSize) {
        generateStep();
    }
    console.log('generated', serias)
}

function generateStep() {
  const lstm1 = (data, c, h) => tf.basicLSTMCell(forgetBias, lstmKernel1, lstmBias1, data, c, h);
  const lstm2 = (data, c, h) => tf.basicLSTMCell(forgetBias, lstmKernel2, lstmBias2, data, c, h);
  const lstm3 = (data, c, h) => tf.basicLSTMCell(forgetBias, lstmKernel3, lstmBias3, data, c, h);

  let outputs = [];
  [c, h, outputs] = tf.tidy(() => {
    // Generate some notes.
    const innerOuts = [];
    for (let i = 0; i < STEPS_PER_GENERATE_CALL; i++) {
      // Use last sampled output as the next input.
      const eventInput = tf.oneHot(lastSample.as1D(), EVENT_SIZE).as1D();
      // Dispose the last sample from the previous generate call, since we
      // kept it.
      if (i === 0) {
        lastSample.dispose();
      }
      const conditioning = getConditioning();
      const axis = 0;
      const input = conditioning.concat(eventInput, axis).toFloat();
      const output = tf.multiRNNCell([lstm1, lstm2, lstm3], input.as2D(1, -1), c, h);

      c.forEach(c => c.dispose());
      h.forEach(h => h.dispose());


      c = output[0];
      h = output[1];

      const outputH = h[2];
      const logits = outputH.matMul(fcW).add(fcB);

      const sampledOutput = tf.multinomial(logits.as1D(), 1).asScalar();

      innerOuts.push(sampledOutput);
      lastSample = sampledOutput;
    }
    return [c, h, innerOuts];
  });

  for (let i = 0; i < outputs.length; i++) {
    playOutput(outputs[i].dataSync()[0])
  }
}


const notes = ['c', 'cs', 'd', 'ds', 'e', 'f', 'fs', 'g', 'gs', 'a', 'as', 'h'];

const NOTES_PER_OCTAVE = 12;
const DENSITY_BIN_RANGES = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0];
const PITCH_HISTOGRAM_SIZE = NOTES_PER_OCTAVE;

let pitchHistogram = [1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0]
function updateConditioningParams() {
  noteDensityEncoding?.dispose();
  noteDensityEncoding = undefined;


  const noteDensityIdx = 0;
  noteDensityEncoding = tf.oneHot(tf.tensor1d([noteDensityIdx + 1], 'int32'), DENSITY_BIN_RANGES.length + 1).as1D();

  pitchHistogramEncoding?.dispose();
  pitchHistogramEncoding = undefined;

  const buffer = tf.buffer([PITCH_HISTOGRAM_SIZE], 'float32');
  const pitchHistogramTotal = pitchHistogram.reduce((prev, val) => prev + val);
  for (let i = 0; i < PITCH_HISTOGRAM_SIZE; i++) {
    buffer.set(pitchHistogram[i] / pitchHistogramTotal, i);
  }
  pitchHistogramEncoding = buffer.toTensor();
}
let partSize = 32;
let playOutPos = 0;
let serias = new Float32Array(partSize).map((v) => NaN);
function playOutput(index) {
    let offset = 0;
    console.log(index);
    for (const eventRange of EVENT_RANGES) {
        const eventType = eventRange[0];
        const minValue = eventRange[1];
        const maxValue = eventRange[2];
        if (offset <= index && index <= offset + maxValue - minValue) {
            if (eventType === 'note_on') {
              const noteNum = index - offset;
              console.log('note_on', noteNum, notes[Math.ceil((noteNum/12 % 1) * 10)]);
              serias[playOutPos] = Math.ceil((noteNum/12 % 1) * 10);
              return
            } else if (eventType === 'note_off') {
              const noteNum = index - offset;
              console.log('note_off');
              /* потушить ноту noTeNum?*/
              return;
            } else if (eventType === 'time_shift') {
              console.log('time_shift');
              playOutPos++;
              return;
            } else if (eventType === 'velocity_change') {
              console.log('velocity_change');

              return;
            } else {
              throw new Error('Could not decode eventType: ' + eventType);
            }
        }

        offset += maxValue - minValue + 1;
    }
}

let playIndex = 0;
let notesDelays = [4.75, 4.32, 3.86, 3.52, 3.16, 2.824, 2.5, 2.198, 1.910, 1.640, 1.15]
let intervalId = 0;
function play1() {
    console.log(serias)
    const source2 = ctx.createBufferSource();
    source2.buffer = audioBuffer;
    source2.connect(ctx.destination);
    source2.start();
    intervalId = setInterval(playTick, 500);
}

function playTick() {
  const noteIndex = serias[playIndex];
  if (noteIndex === undefined) {
      clearInterval(intervalId);
      intervalId = 0;
      playIndex = 0;
      return
  }
  const delay = notesDelays[noteIndex];
  if (!isNaN(delay)) {
    guitar[0].play(delay / 1000, 400)
  }
  playIndex++;
}

let intervalIDBassAnalyzing;
let intervalBassAnalyzingIndex = 0;
let pitches = [];
function detectPich() {
  if (intervalBassAnalyzingIndex > partSize) {
      clearInterval(intervalIDBassAnalyzing);
      const notes = pitches.filter((v) => v !== -1).map(noteFromPitch)
      pitchHistogram = [0, 0,0,0,0,0,0,0,0,0,0,0];
      notes.forEach((note) => {
          pitchHistogram[note%12] = 2;
          if (pitchHistogram[note%12 + 7]) {
              pitchHistogram[note%12 + 7] = 1
          }
      });
      updateConditioningParams();
      console.log(pitches)
      console.log(pitchHistogram);
      console.log('Bass check complete');
  }
  analyser.getFloatTimeDomainData(arrayData);
  const ac = autoCorrelate(arrayData, ctx.sampleRate);
  console.log(ac)
  pitches.push(ac);
  intervalBassAnalyzingIndex++;
}

function bassAnalysing() {
    pitches = [];

    source.buffer = audioBuffer;
    source.start();

    intervalIDBassAnalyzing = setInterval(() => detectPich(), 500);
    ;
}


let mediaRecorder;
let chunks;
function onSuccess(stream) {
    mediaRecorder = new MediaRecorder(stream);
    chunks = [];
    mediaRecorder.onstop = function(e) {
          console.log("onstop() called.", e);

          var blob = new Blob(chunks, {
            'type': 'audio/wav'
          });
          chunks = [];

          var reader = new FileReader();
          reader.addEventListener("loadend", async function() {
            audioBuffer = await ctx.decodeAudioData(reader.result);
          });
          reader.readAsArrayBuffer(blob);
        }

    mediaRecorder.ondataavailable = function(e) {
      chunks.push(e.data);
    }
}

function askRecord() {
    var constraints = { audio: true };
    navigator.getUserMedia(constraints, onSuccess, ()=>{ console.log('recordError ')});
}

let recordInterval;
let recordIntervalIndex = 0;
function startRecord() {
    recordInterval = setInterval(() => {
        guitar[0].play(8.56 / 1000, 0.01);
        if (recordIntervalIndex === 4) {
             mediaRecorder.start();
        }

        if (recordIntervalIndex === 20) {
            clearInterval(recordInterval);
            mediaRecorder.stop();
        }
        recordIntervalIndex++
    }, 1000)
    //mediaRecorder.start();
}
document.getElementById('load').addEventListener('click', start);
document.getElementById('askRecord').addEventListener('click', askRecord);
document.getElementById('record').addEventListener('click', startRecord);
document.getElementById('play1').addEventListener('click', play1);
document.getElementById('generate').addEventListener('click', generate);
document.getElementById('bassAnalysing').addEventListener('click', bassAnalysing);

