// Load .env and merge into existing app.json so all settings are preserved
const fs = require('fs');
const path = require('path');

function loadEnv(dotenvPath) {
  try {
    const full = fs.readFileSync(dotenvPath, 'utf8');
    const lines = full.split(/\r?\n/);
    const out = {};
    for (const line of lines) {
      const m = line.match(/^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$/);
      if (!m) continue;
      let [, k, v] = m;
      v = v.replace(/^\"|\"$/g, '').replace(/^'|'$/g, '');
      out[k] = v;
    }
    return out;
  } catch {
    return {};
  }
}

module.exports = () => {
  const envPath = path.join(__dirname, '.env');
  const env = loadEnv(envPath);

  // Read base config from app.json
  const appJsonPath = path.join(__dirname, 'app.json');
  const base = fs.existsSync(appJsonPath)
    ? JSON.parse(fs.readFileSync(appJsonPath, 'utf8'))
    : { expo: {} };

  const expo = base.expo || {};
  expo.extra = {
    ...(expo.extra || {}),
    API_BASE: env.API_BASE || expo.extra?.API_BASE || 'http://localhost:8000',
  };

  return { expo };
};
