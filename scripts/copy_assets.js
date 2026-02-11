const fs = require('fs');
const path = require('path');

const assets = [
    { src: 'node_modules/alpinejs/dist/cdn.min.js', dest: 'src/parsfet/static/js/alpine.min.js' },
    { src: 'node_modules/plotly.js-dist-min/plotly.min.js', dest: 'src/parsfet/static/js/plotly.min.js' }
];

assets.forEach(asset => {
    const destDir = path.dirname(asset.dest);
    if (!fs.existsSync(destDir)) {
        fs.mkdirSync(destDir, { recursive: true });
    }
    fs.copyFileSync(asset.src, asset.dest);
    console.log(`Copied ${asset.src} to ${asset.dest}`);
});
