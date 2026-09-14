'use strict';
const fs = require('fs');
const path = require('path');
function loadFrom(file, id, name){
  const html = fs.readFileSync(file, 'utf8');
  const m = html.match(new RegExp('<script id="' + id + '">([\\s\\S]*?)<\\/script>'));
  if(!m) throw new Error(id + ' script not found in ' + file);
  return (0, eval)(m[1] + '\n;' + name);
}
module.exports = function loadCore(){ return loadFrom(path.join(__dirname, '..', 'index.html'), 'ms-core', 'MS'); };
module.exports.loadFrom = loadFrom;
