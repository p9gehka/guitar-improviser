import { MetalString } from './helpers-link/guitar/key3.5.js';
import { Guitar } from './helpers-link/guitar/guitar.js';
import { autoCorrelate, noteFromPitch } from './helpers-link/auto-correlate.js';
import { CheckpointLoader } from './checkpoint-loader.js';

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
let performanceGainNode;
let mainGainNode;

async function load() {
    ctx = new AudioContext({ sampleRate: 44100 });
    guitar = new Guitar(ctx);
    const comperessorNode = ctx.createDynamicsCompressor();
    performanceGainNode = ctx.createGain();
    performanceGainNode.gain.setValueAtTime(0.02, ctx.currentTime)
    await guitar.connect(performanceGainNode);

    performanceGainNode.connect(comperessorNode);
    comperessorNode.connect(ctx.destination);

    analyser = ctx.createAnalyser();
    analyser.fftSize = 2048;
    source = ctx.createBufferSource();

    mainGainNode = ctx.createGain();

    arrayData = new Float32Array(2048);
    source.connect(analyser);
    analyser.connect(mainGainNode)
    mainGainNode.connect(ctx.destination)
    const resp = await fetch("./sources-link/7army.m4a");
    const arrayBuffer = await resp.arrayBuffer();
    audioBuffer = await ctx.decodeAudioData(arrayBuffer);

    const vars  = await new CheckpointLoader().getAllVariables();

    lstmKernel1 = vars['rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel'];
    lstmBias1 = vars['rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias'];
    lstmKernel2 = vars['rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel'];
    lstmBias2 = vars['rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias'];
    lstmKernel3 = vars['rnn/multi_rnn_cell/cell_2/basic_lstm_cell/kernel'];
    lstmBias3 = vars['rnn/multi_rnn_cell/cell_2/basic_lstm_cell/bias'];

    fcB = vars['fully_connected/biases'];
    fcW = vars['fully_connected/weights'];
    resetRnn();
    askRecord();
    console.log('loaded')
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

async function generate() {
    updateConditioningParams();
    playOutPos = 0;
    serias = new Float32Array(partSize).map((v) => NaN);
    while (playOutPos < partSize) {
        generateStep();
    }

    console.log('generated', serias, velocitySerias, effectSerias)
    return {
      serias,
      velocitySerias,
      effectSerias
    }
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


const notes = [
  'C', 'Cs', 'D', 'Ds', 'E', 'F', 'Fs', 'G', 'Gs', 'A', 'As', 'H',
  'c', 'cs', 'd', 'ds', 'e', 'f', 'fs', 'g', 'gs', 'a', 'as', 'h'
 ];

const NOTES_PER_OCTAVE = 12;
const DENSITY_BIN_RANGES = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0];
const PITCH_HISTOGRAM_SIZE = NOTES_PER_OCTAVE;

let pitchHistogram = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
function updateConditioningParams() {
  noteDensityEncoding?.dispose();
  noteDensityEncoding = undefined;
  if (pitchHistogram.filter(v=>v !== 0).length === 0) {
    pitchHistogram = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1];
  }

  const noteDensityIdx = 5;
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

const barSize = 32;
const barTime = 8000;

const  partSize = barSize * 8;
const emptyPartSize = barSize *4;
let playOutPos = 0;
let serias = new Float32Array(partSize).map((v) => NaN);
let velocitySerias = new Float32Array(partSize).map(v=>0);
let effectSerias = new Float32Array(partSize).map(v=>0);
let currentVelocity = 0.4;
let slide = 0
function playOutput(index) {
    let offset = 0;
    for (const eventRange of EVENT_RANGES) {
        const eventType = eventRange[0];
        const minValue = eventRange[1];
        const maxValue = eventRange[2];
        if (offset <= index && index <= offset + maxValue - minValue) {
            if (eventType === 'note_on') {
              const noteNum = index - offset;
              serias[playOutPos] = Math.round(noteNum/24 % 1 * 24);
              velocitySerias[playOutPos] = currentVelocity;
              effectSerias[playOutPos] = slide;
              slide = 0
              return
            } else if (eventType === 'note_off') {

              playOutPos++;
              //console.log('note_off', noteNum);
              /* потушить ноту noTeNum?*/
              return;
            } else if (eventType === 'time_shift') {
              const noteNum = index - offset;
              if (noteNum%15 === 0) {
                /* temporary */
                 slide = 1
              }
              
              return;
            } else if (eventType === 'velocity_change') {
              
              currentVelocity = (index - offset + 1) * Math.ceil(127 / VELOCITY_BINS);
              currentVelocity = currentVelocity / 127;
              return;
            } else {
              throw new Error('Could not decode eventType: ' + eventType);
            }
        }

        offset += maxValue - minValue + 1;
    }
}

let playIndex = 0;
let intervalId = 0;
function play1() {
    console.log(serias)
    const playSource = ctx.createBufferSource();
    playSource.buffer = audioBuffer;
    playSource.connect(ctx.destination);
    playSource.start();
    playTick();
    intervalId = setInterval(playTick, barTime / barSize);
}

function playTick() {
  const noteIndex = serias[playIndex];
  const velocity = velocitySerias[playIndex];
  const slide = effectSerias[playIndex];
  if (noteIndex === undefined) {
      clearInterval(intervalId);
      intervalId = 0;
      playIndex = 0;
      return
  }
  const note = notes[noteIndex];
  if (note !== undefined) {
    guitar.play(note, 16 / barSize, velocity, slide === 1);
  }
  playIndex++;
}

let intervalIDBassAnalyzing;
let intervalBassAnalyzingIndex = 0;
let pitches = [];

function detectPich() {
  if (intervalBassAnalyzingIndex > 32) {
      clearInterval(intervalIDBassAnalyzing);
      const newNotes = pitches.filter((v) => v !== -1).map(noteFromPitch)
      pitchHistogram = [0,0,0,0,0,0,0,0,0,0,0,0];
      newNotes.forEach((note) => {
          pitchHistogram[note%12] = 2;
          if (pitchHistogram[note%12 + 7]) {
              pitchHistogram[note%12 + 7] = 1
          }
      });

      console.log(pitchHistogram);
      console.log('Bass check complete');
  }
  /* надо определять ноту чуть позже такта */
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

    intervalIDBassAnalyzing = setInterval(() => detectPich(), 250);
}

let mediaRecorder;
let chunks = [];
function onSuccess(stream) {
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.onstop = mediaRecorderStop
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
        guitar.play('c', 0.01);
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


let tickIntervalID;
let tickIndex = 0;

let tickSize = 250;
let barTickNumber = 16;
let barDuration = tickSize * barTickNumber;

let preRecordTickNumbers = 2 * barTickNumber;

let counterStart = 1 * barTickNumber;
let counterTicksNumbers = 1 * barTickNumber;

let recordTickNumbers = 3 * barTickNumber;
let recordTickStart = preRecordTickNumbers


let metr = 4;
let audioProcessed = false;

let analysingStartedIndex;
let analysingCompleted = false;
let pitchSaved = false;
let performanceIndex = [undefined, undefined];
const results = [];
let generationComplete = [false, false];
let performanceStart = [NaN, NaN];
let generateID = 0;
let generateIndex;
let firstGeneration = true;
function startRecord2() {
    tickIntervalID = setInterval(() => {
        const isMetrPoint = tickIndex % metr === 0;
        if (isMetrPoint && source.buffer === null) {
           guitar.play('c', 0.01);
        }

        if (isMetrPoint && counterStart < tickIndex && tickIndex < (counterStart + counterTicksNumbers)) {
          console.log(metr - ((tickIndex - counterStart) / metr));
        }

        if (preRecordTickNumbers === tickIndex) {
             console.log('Record Started', tickIndex);
             mediaRecorder.start();
        }


        if ((recordTickStart + recordTickNumbers) === tickIndex) {
            console.log('Record completed loading', tickIndex);
            mediaRecorder.stop();
        }

        if(audioProcessed && source.buffer === null) {
           pitches = [];
           source.loop = true;
           source.loopEnd = 16;
           source.buffer = audioBuffer;
           source.start();
        }

        if(source.buffer !== null && !analysingCompleted) {
          if (analysingStartedIndex === undefined) {
             console.log('Analysing', tickIndex);
             analysingStartedIndex = tickIndex;
          }
          detectPich2();
          if (tickIndex - analysingStartedIndex === recordTickNumbers) {
            analysingCompleted = true
          }
        }

        if (analysingCompleted && !pitchSaved) {
          const newNotes = pitches.filter((v) => v !== -1).map(noteFromPitch)
          pitchHistogram = [0,0,0,0,0,0,0,0,0,0,0,0];
          newNotes.forEach((note) => {
              pitchHistogram[note%12] = 2;
              if (pitchHistogram[note%12 + 7]) {
                  pitchHistogram[note%12 + 7] = 1
              }
          });
          console.log('Analysing completed', pitchHistogram);
          pitchSaved = true;
           (async (identifire) => {
             results[identifire] = await generate();
             generationComplete[identifire] = true;
             console.log(`generationComplete${identifire}`);
          })(0)
        }

        if (performanceStart[1] + 1 === tickIndex) {
          (async (identifire) => {
             results[identifire] = await generate();
             generationComplete[identifire] = true;
             console.log(`generationComplete${identifire}`);
          })(0)
        }

        if (generationComplete[0] && firstGeneration) {
          performanceStart[0] = Math.ceil(tickIndex / barTickNumber) * barTickNumber
          firstGeneration = false;
        }

        if (performanceStart[0] <= tickIndex && tickIndex < partSize + performanceStart[0]) {
           playTick2(0, tickIndex - performanceStart[0]);
        }

        if (performanceStart[0] + 1 === tickIndex) {
          (async (identifire) => {
            results[identifire] = await generate()
            generationComplete[identifire] = true;
            console.log(`generationComplete${identifire}`);
          })(1)
        }

        if (partSize + performanceStart[0] === tickIndex) {
          performanceStart[1] = partSize + emptyPartSize + performanceStart[0]
          console.log('performance0 Ended');
        }

        if (performanceStart[1] <= tickIndex && tickIndex < partSize + performanceStart[1]) {
           playTick2(1, tickIndex - performanceStart[1]);
        }

        if (partSize + performanceStart[1] === tickIndex) {
          performanceStart[0] = partSize + emptyPartSize + performanceStart[1];
          console.log('performance1 Ended');
        }
        tickIndex++;
    }, tickSize)
}

function playTick2(resultIndex, stepIndex) {
  const noteIndex = results[resultIndex].serias[stepIndex];
  const velocity = results[resultIndex].velocitySerias[stepIndex];
  const slide = results[resultIndex].effectSerias[stepIndex];
  const note = notes[noteIndex];
  if (note !== undefined) {
    guitar.play(note, 16 / barSize, velocity, slide === 1);
  }
}


function detectPich2() {
  analyser.getFloatTimeDomainData(arrayData);
  const ac = autoCorrelate(arrayData, ctx.sampleRate);
  console.log(ac)
  pitches.push(ac);
}


function mediaRecorderStop(e) {
    console.log("onstop() called.", e);
    var blob = new Blob(chunks, { 'type': 'audio/wav' });
    chunks = [];
    var reader = new FileReader();
    reader.addEventListener("loadend", async function() {
      audioBuffer = await ctx.decodeAudioData(reader.result);
      audioProcessed = true;
    });
    reader.readAsArrayBuffer(blob);
}

function start() {
  console.log('Ready?');
  startRecord2();
}
function stop() {
  clearInterval(tickIntervalID);
  source.stop();
}

function mainGainChange(e) {
   mainGainNode?.gain.setValueAtTime(Number(e.target.value), ctx.currentTime)
}
function performanceGainChange(e) {
  performanceGainNode?.gain.setValueAtTime(Number(e.target.value), ctx.currentTime)
}

document.getElementById('load').addEventListener('click', load);
document.getElementById('askRecord').addEventListener('click', askRecord);
document.getElementById('record').addEventListener('click', startRecord);
document.getElementById('play1').addEventListener('click', play1);
document.getElementById('generate').addEventListener('click', generate);
document.getElementById('bassAnalysing').addEventListener('click', bassAnalysing);
document.getElementById('start').addEventListener('click', start);
document.getElementById('stop').addEventListener('click', stop);

document.getElementById('main').addEventListener('change', mainGainChange);
document.getElementById('performance').addEventListener('change', performanceGainChange);

