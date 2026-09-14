'use strict';
const fs = require('fs');
const path = require('path');
module.exports = function loadCore(){
  const html = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');
  const m = html.match(/<script id="lb-core">([\s\S]*?)<\/script>/);
  if(!m) throw new Error('lb-core script not found');
  return (0, eval)(m[1] + '\n;LB');
};
