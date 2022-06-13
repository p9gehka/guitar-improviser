const CHECKPOINT_URL = '/rnn-link'
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
          var _this = this;
          const result = await fetch(_this.urlPath + MANIFEST_FILE);
          _this.checkpointManifest = await result.json()
          return;
          /*
          return new Promise(function (resolve, reject) {
              var xhr = new XMLHttpRequest();
              xhr.open('GET', _this.urlPath + MANIFEST_FILE);
              xhr.onload = function () {
                  _this.checkpointManifest = JSON.parse(xhr.responseText);
                  resolve();
              };
              xhr.onerror = function (error) {
                  throw new Error(MANIFEST_FILE + " not found at " + _this.urlPath + ". " + error);
              };
              xhr.send();
          });
          */
      }

   getCheckpointManifest() {
           var _this = this;
           if (this.checkpointManifest == null) {
               return new Promise(function (resolve, reject) {
                   _this.loadManifest().then(function () {
                       resolve(_this.checkpointManifest);
                   });
               });
           }
           return new Promise(function (resolve, reject) {
               resolve(_this.checkpointManifest);
           });
       }
   getAllVariables () {
           var _this = this;
           if (this.variables != null) {
               return new Promise(function (resolve, reject) {
                   resolve(_this.variables);
               });
           }
           return new Promise(function (resolve, reject) {
               _this.getCheckpointManifest().then(function (checkpointDefinition) {
                   var variableNames = Object.keys(_this.checkpointManifest);
                   var variablePromises = [];
                   for (var i = 0; i < variableNames.length; i++) {
                       variablePromises.push(_this.getVariable(variableNames[i]));
                   }
                   Promise.all(variablePromises).then(function (variables) {
                       _this.variables = {};
                       for (var i = 0; i < variables.length; i++) {
                           _this.variables[variableNames[i]] = variables[i];
                       }
                       resolve(_this.variables);
                   });
               });
           });
       }

   getVariable(varName) {
           var _this = this;
           if (!(varName in this.checkpointManifest)) {
               throw new Error('Cannot load non-existant variable ' + varName);
           }
           var variableRequestPromiseMethod = function (resolve, reject) {
               var xhr = new XMLHttpRequest();
               xhr.responseType = 'arraybuffer';
               var fname = _this.checkpointManifest[varName].filename;
               xhr.open('GET', _this.urlPath + fname);
               xhr.onload = function () {
                   if (xhr.status === 404) {
                       throw new Error("Not found variable " + varName);
                   }
                   var values = new Float32Array(xhr.response);

                   var tensor = tf.tensor(values, _this.checkpointManifest[varName].shape);
                   resolve(tensor);
               };
               xhr.onerror = function (error) {
                   throw new Error("Could not fetch variable " + varName + ": " + error);
               };
               xhr.send();
           };
           if (this.checkpointManifest == null) {
               return new Promise(function (resolve, reject) {
                   _this.loadManifest().then(function () {
                       new Promise(variableRequestPromiseMethod).then(resolve);
                   });
               });
           }
           return new Promise(variableRequestPromiseMethod);
       };
}