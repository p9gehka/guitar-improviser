"use strict";

class PitchDetector extends AudioWorkletProcessor {
    _savedList = new Float32Array(2048).fill(0);
    _listSize = 2048;
    _counter = -1
    constructor() {
        super();
        this.port.onmessage = () => {
            this._counter = currentFrame + this._listSize;
        };
    }


    process([input], [output], params) {
        if (this._counter > 0) {
            this._savedList.copyWithin(input[0].length);
            this._savedList.set(input[0])
        }
        if (currentFrame > this._counter && this._counter > 0) {
            this._counter = -1;
            console.log(this.autoCorrelate(this._savedList))
        }
        return true;
    }

    autoCorrelate(buf) {
        // Implements the ACF2+ algorithm
        let SIZE = buf.length;
        let power = 0
        for (let i=0;i<SIZE;i++) {
            power += buf[i]**2;
        }

        const rms = Math.sqrt(power/SIZE);
        if (rms<0.01) // not enough signal
            return -1;

         /* Trimming cuts the edges of the signal so that it starts and ends near zero. 
        This is used to neutralize an inherent instability of the ACF version I use.*/

        var r1=0, r2=SIZE-1, thres=0.2;
        for (var i=0; i<SIZE/2; i++)
            if (Math.abs(buf[i])<thres) { r1=i; break; }
        for (var i=1; i<SIZE/2; i++)
            if (Math.abs(buf[SIZE-i])<thres) { r2=SIZE-i; break; }

        buf = buf.slice(r1,r2);
        SIZE = buf.length;
        [1,2,3]
        var c = new Array(SIZE).fill(0);
        for (var i=0; i<SIZE; i++)    
            for (var j=0; j<SIZE-i; j++)
                c[i] = c[i] + buf[j]*buf[j+i];

        var d=0; while (c[d]>c[d+1]) d++;
        var maxval=-1, maxpos=-1;
        for (var i=d; i<SIZE; i++) {
            if (c[i] > maxval) {
                maxval = c[i];
                maxpos = i;
            }
        }
        var T0 = maxpos;

        /* Interpolation is parabolic interpolation. It helps with precision. 
             We suppose that a parabola pass through the three points that comprise the peak. 
             'a' and 'b' are the unknowns from the linear equation system 
             and b/(2a) is the "error" in the abscissa. 
             y1,y2,y3 are the ordinates.*/
        var x1=c[T0-1], x2=c[T0], x3=c[T0+1];
        let a = (x1 + x3 - 2*x2)/2;
        let b = (x3 - x1)/2;
        if (a) T0 = T0 - b/(2*a);

        return sampleRate/T0;
    }
}

registerProcessor("pitch-detector", PitchDetector);