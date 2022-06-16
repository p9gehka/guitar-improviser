export class MetalString {
	constructor(context, pan) {
		this.context = context;
		this.gainNode = context.createGain();
		this.gainNode.gain.setValueAtTime(0, context.currentTime);
		this.delayNode = context.createDelay();
		this.panNode = this.context.createStereoPanner();
		this.panNode.pan.setValueAtTime(pan, context.currentTime);
		this.noiseGeneratorGainNode = context.createGain();
		this.noiseGeneratorGainNode.gain.setValueAtTime(1, context.currentTime);

		this.endLivingTime = 0;
		this.slidingEnd = 0;
	}

	async connect(dest) {
		await this.context.audioWorklet.addModule('helpers-link/guitar/small-noise.js');
		this.noiseGenerator = new AudioWorkletNode(this.context, 'small-noise');

		const splitterNode = this.context.createChannelSplitter();
		const mergerNode = this.context.createChannelMerger();

		const filter1 = this.context.createBiquadFilter();
		const filter2 = this.context.createBiquadFilter();

		filter1.frequency.setValueAtTime(20050, this.context.currentTime);
		filter1.Q.setValueAtTime(4, this.context.currentTime);

		filter2.frequency.setValueAtTime(2100, this.context.currentTime);
		filter2.Q.setValueAtTime(2, this.context.currentTime);


		this.noiseGenerator.connect(this.noiseGeneratorGainNode);
		this.noiseGeneratorGainNode.connect(filter1)
		filter1.connect(filter2);
		filter2.connect(mergerNode);
		mergerNode.connect(this.delayNode);
		this.delayNode.connect(splitterNode);
		splitterNode.connect(this.gainNode);
		this.gainNode.connect(mergerNode);
		splitterNode.connect(this.panNode);
		this.panNode.connect(dest);
	}
	slide(pitches, duration) {
		const { currentTime } = this.context;
		const realDuration = Math.min(this.endLivingTime - currentTime - 0.05, duration);
		this.delayNode.delayTime.setValueCurveAtTime(pitches, currentTime, realDuration);
		this.slidingEnd = currentTime + realDuration;
	}
	play(pitch, duration, volume = 1) {
		const { currentTime } = this.context;
		const { gain } = this.gainNode;
		this.noiseGeneratorGainNode.gain.setValueAtTime(volume, currentTime);
		gain.setValueAtTime(0.987 + volume / 200, currentTime); /* 75 93 */
		this.endLivingTime = currentTime + duration - 0.01;
		gain.setTargetAtTime(0, this.endLivingTime, 0.05);

		this.delayNode.delayTime.setValueAtTime(pitch, currentTime);
		this.noiseGenerator?.port.postMessage({});
	}

	sliding() {
		return this.slidingEnd + 0.05 >= this.context.currentTime;
	}
	ringing() {
		return this.endLivingTime - this.context.currentTime >= 0;
	}
}