const fs = require('fs');
const os = require('os');
const path = require('path');

const [input, output, width, height] = process.argv.slice(2);
if (!input || !output) throw new Error('Usage: node render_png.js input.svg output.png [width] [height]');

let sharp;
try {
  sharp = require('sharp');
} catch (_) {
  const bundled = path.join(os.homedir(), '.cache', 'codex-runtimes', 'codex-primary-runtime',
    'dependencies', 'node', 'node_modules', 'sharp', 'dist', 'index.cjs');
  sharp = require(process.env.CODEX_SHARP_PATH || bundled);
}

sharp(fs.readFileSync(input), { density: 300 })
  .resize(Number(width || 2100), Number(height || 1400))
  .png({ compressionLevel: 9 })
  .withMetadata({ density: 300 })
  .toFile(output);

