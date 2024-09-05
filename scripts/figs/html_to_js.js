const puppeteer = require('puppeteer');
const path = require('path');

async function htmlToPdf(htmlFilePath, outputPdfPath, aspectRatio = 9/9) {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();
  
  // Set the viewport to match the desired aspect ratio
  // For example, for 16:9, we could use 1600x900
  const viewportWidth = 750;
  const viewportHeight = Math.round(viewportWidth / aspectRatio);
  
  await page.setViewport({
    width: viewportWidth,
    height: viewportHeight
  });

  // Load the HTML file
  await page.goto(`file:${path.resolve(htmlFilePath)}`, {
    waitUntil: 'networkidle0'
  });

  // Generate PDF
  await page.pdf({
    path: outputPdfPath,
    width: `${viewportWidth}px`,
    height: `${viewportHeight}px`,
    printBackground: true
  });

  await browser.close();
  console.log(`PDF saved to ${outputPdfPath}`);
}

// Usage
htmlToPdf('scripts/figs/training_vertical.html', 'output.pdf', 9/9);