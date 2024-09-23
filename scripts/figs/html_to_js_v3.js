const puppeteer = require('puppeteer');
const path = require('path');

async function htmlToPdf(htmlFilePath, outputPdfPath) {
  const browser = await puppeteer.launch({ headless: "new" });
  const page = await browser.newPage();

  // Set a larger initial viewport
  await page.setViewport({ width: 800, height: 225 });

  // Navigate to the HTML file
  await page.goto(`file:${path.resolve(htmlFilePath)}`, { waitUntil: 'networkidle0' });

  // Get the full page dimensions
  const dimensions = await page.evaluate(() => {
    return {
      width: Math.max(document.documentElement.scrollWidth, document.body.scrollWidth),
      height: Math.max(document.documentElement.scrollHeight, document.body.scrollHeight)
    };
  });

  console.log('Page dimensions:', dimensions);

  // Add some padding
  const padding = 0;
  // dimensions.width += padding * 2;
  // dimensions.height += padding * 2;

  // Set viewport to match content size
  await page.setViewport(dimensions);

  // Take a full page screenshot for comparison
  await page.screenshot({ path: 'full_page_screenshot.png', fullPage: true });

  // Generate PDF
  await page.pdf({
    path: outputPdfPath,
    width: dimensions.width,
    height: dimensions.height,
    printBackground: true,
    margin: { top: padding, right: padding, bottom: padding, left: padding }
  });

  await browser.close();
  console.log(`PDF saved to ${outputPdfPath} with dimensions ${dimensions.width}x${dimensions.height}`);
  console.log('Full page screenshot saved to full_page_screenshot.png for comparison');
}

// Usage
htmlToPdf('scripts/figs/pipeline_3_steps_v3.html', 'pipeline.pdf')
  .catch(error => console.error('Error:', error));
