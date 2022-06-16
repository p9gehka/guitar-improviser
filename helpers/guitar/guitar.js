import { MetalString } from './key3.5.js';
import { delays, notesOrdLow, notesOrdHight, noteTranspose } from './delays.js';

export class GuitarString {
	_pull = [];
	_lastString = null;
	_lastNote = null;
	_notesOrd = [];
	constructor(context, pullSize) {
		this._pull = Array.from(new Float32Array(pullSize)).map(() => new MetalString(context, 0));
		this.context = context;
	}

	async connect(dest){
		const mergerNode = this.context.createChannelMerger();
		const splitterNode = this.context.createChannelSplitter();
		await Promise.all(this._pull.map(key => key.connect(mergerNode)));
		mergerNode.connect(splitterNode)
		splitterNode.connect(dest)
		return
	}

	_getMetalStringFromPull() {
		for (let st of this._pull) {
			if(!st.ringing()) {
				return st
			}
		}
		return null
	}

	play(note, duration, volume, slide = false) {
		const delayValue = this.getDelay(note);

		if (isNaN(delayValue)) {
			return;
		}

		const lastNote = this._lastNote
		const currentNoteIndex = this._notesOrd.findIndex((v) => v === note);
		if (slide && this._lastString?.ringing() && !this._lastString?.sliding() && lastNote !== note && currentNoteIndex !== -1) {
			const lastNoteIndex = this._notesOrd.findIndex((v) => v === lastNote);
			const first = Math.min(currentNoteIndex, lastNoteIndex)
			const last = Math.max(currentNoteIndex, lastNoteIndex)
			const slideSequence = this._notesOrd.slice(first, last);
			if (lastNoteIndex > currentNoteIndex) {
				slideSequence.reverse();
			}
			const maxSlideDuration = slideSequence.length / 2;
			const doubledSequence = slideSequence.reduce(
				(arr,v)=>[...arr, this.getDelay(v), this.getDelay(v), this.getDelay(v), this.getDelay(v)], []
			);
			this._lastString?.slide(doubledSequence, maxSlideDuration);
			this._lastNote = note;
			return;
		}
		this._lastString = this._getMetalStringFromPull()

		const preNote = this._notesOrd[currentNoteIndex+1];
		if(slide && preNote !== undefined && preNote !== -1) {
			const preDelay = this.getDelay(preNote);
			this._lastString?.play(preDelay, duration, volume)
			this._lastString?.slide([preDelay, delayValue], 0.5);

			this._lastNote = note;
			return
		}
		
		this._lastString?.play(delayValue, duration, volume)
		this._lastNote = note;
	}

	getDelay(note) {
		return (delays[note] + (Math.random() / 100)) / 1000;
	}
}

class GuitarHightString extends GuitarString {
	constructor(context) {
		super(context, 5);
		this._notesOrd = notesOrdHight;
	}
	async connect(dest) {
		await this.context.audioWorklet.addModule('helpers-link/guitar/phase-vocoder.js');
		this.phaseVocoderNode = new AudioWorkletNode(this.context, 'phase-vocoder-processor');
		this.phaseVocoderNode.parameters.get('pitchFactor').value = 2;
		super.connect(this.phaseVocoderNode);
		this.phaseVocoderNode.connect(dest);
	}
}
class GuitarLowString extends GuitarString {
	constructor(context) {
		super(context, 20);
		this._notesOrd = notesOrdLow;
	}
}

export class Guitar {
	constructor(context) {
		this._guitarLow = new GuitarLowString(context);
		this._guitarHight = new GuitarHightString(context);
		this.context = context;
	}
	async connect(dest) {
		const mergerNode = this.context.createChannelMerger();
		const splitterNode = this.context.createChannelSplitter();
		await this._guitarLow.connect(mergerNode);
		await this._guitarHight.connect(mergerNode);
		mergerNode.connect(splitterNode)
		splitterNode.connect(dest)
	}

	play(note, duration, volume, slide = false) {
		if (note in noteTranspose) {
			this._guitarHight.play(noteTranspose[note], duration, volume, slide)
		} else {
			this._guitarLow.play(note, duration, volume, slide)
		}
	}
}
