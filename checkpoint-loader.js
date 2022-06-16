const CHECKPOINT_URL = '/rnn'
//const CHECKPOINT_URL = 'https://storage.googleapis.com/download.magenta.tensorflow.org/models/performance_rnn/tfjs';
const MANIFEST_FILE = 'manifest.json';
export class CheckpointLoader {
  constructor() {
    this.urlPath = CHECKPOINT_URL;
    if (this.urlPath.charAt(this.urlPath.length - 1) !== '/') {
         this.urlPath += '/';
    }
  }
  async loadManifest() {
      const result = await fetch(this.urlPath + MANIFEST_FILE);
      this.checkpointManifest = await result.json()
      return this.checkpointManifest;
  }

   async getAllVariables () {
       const manifest = await this.loadManifest()

       const variableNames = Object.keys(manifest);
       const variablePromises = [];
       for (var i = 0; i < variableNames.length; i++) {
           variablePromises.push(this.getVariable(variableNames[i]));
       }
       const variables = await Promise.all(variablePromises)
       this.variables = {};
       for (var i = 0; i < variables.length; i++) {
           this.variables[variableNames[i]] = variables[i];
       }
       return this.variables;
   }

   async getVariable(varName) {
       const fname = this.checkpointManifest[varName].filename;
       const resp = await fetch(this.urlPath + fname)
       const respArrayBuffer = await resp.arrayBuffer();

       const values = new Float32Array(respArrayBuffer);
       return tf.tensor(values, this.checkpointManifest[varName].shape);
   };
}