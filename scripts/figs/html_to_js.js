const puppeteer = require('puppeteer');
const path = require('path');

async function htmlToPdf(htmlFilePath, outputPdfPath) {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();

  // Load the HTML file
  await page.goto(`file:${path.resolve(htmlFilePath)}`, {
    waitUntil: 'networkidle0'
  });

  // Get the exact content size
  const contentSize = await page.evaluate(() => {
    const body = document.body;
    const html = document.documentElement;

    const height = Math.max(
      body.scrollHeight, body.offsetHeight,
      html.clientHeight, html.scrollHeight, html.offsetHeight
    );
    const width = Math.max(
      body.scrollWidth, body.offsetWidth,
      html.clientWidth, html.scrollWidth, html.offsetWidth
    );

    return { width, height };
  });

  // Add a small padding
  const padding = 0;
  contentSize.height -= 40;

  // Set viewport to match content size
  await page.setViewport(contentSize);

  // Generate PDF with exact size
  await page.pdf({
    path: outputPdfPath,
    width: contentSize.width,
    height: contentSize.height,
    printBackground: true,
    margin: { top: padding, right: padding, bottom: padding, left: padding }
  });

  await browser.close();
  console.log(`PDF saved to ${outputPdfPath} with dimensions ${contentSize.width}x${contentSize.height}`);
}

// Usage
htmlToPdf('scripts/figs/training_vertical_6sep.html', 'intentional_shift_training.pdf');
