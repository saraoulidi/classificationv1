const puppeteer = require("puppeteer");
const fs = require("fs");
var glob = require("glob");

brands = glob.sync("keywords/*.txt");

(async () => {
  const browser = await puppeteer.launch({ headless: false });
  const page = await browser.newPage();
  await page.setViewport({ width: 1600, height: 1200 });

  for (let x = 0; x < brands.length; x++) {
    // grab the brand name from the keyword file without the .txt
    let currentBrandName = brands[x].split("/")[1].split(".")[0];

    console.log(`Doing brand ${currentBrandName}`);

    // Read the keyword file
    var buffer = fs.readFileSync(brands[x]);

    // Convert the file buffer to a string and split it line by line
    const keyWords = buffer.toString().split("\n");

    let urls = [];

    // Loop over all the keywords in the array
    for (let i = 0; i < keyWords.length; i++) {
      // Print the current keyword
      console.log(`Doing Keyword: ${keyWords[i]} #${i + 1}/${keyWords.length}`);

      // Go to google image and wait for the page to load.
      await page.goto("https://images.search.yahoo.com/", {
        waitUntil: "load"
      });

      // Wait for the search field to show up.
      await page.waitForSelector("input[name='p']");

      // Type the keyword value into the search field at a speed of 300ms between each keystroke
      await page.type("input[name='p']", keyWords[i], { delay: 100 });

      // Submit the seach form
      await page.click("input[type='submit']", { delay: 150 });

      await page.waitFor(5000);

      // Wait for search results to show up.
      await page.waitForSelector("#res-cont");

      // Scrap urls from images.
      await page.waitForSelector("#resitem-0");
      await page.click("#resitem-0");

      let badURLs = 0;

      for (let j = 0; j < 200; j++) {
        let image_url = await page.evaluate(() => {
          if (document.querySelector("img#img")) {
            return document.querySelector("img#img").src;
          } else {
            return null;
          }
        });

        await page.waitFor(3000);

        try {
          await page.click("button[title='Next Image']");
        } catch (error) {
          await page.click("button[title='Next Image']");
        }

        // only add images with an url, escape base64 and gstatic.com images
        if (
          image_url &&
          image_url.indexOf("data:image/") === -1 &&
          image_url.indexOf("gstatic.com") === -1
        ) {
          console.log(image_url);
          urls.push(image_url);
        } else {
          badURLs++;
          console.log(`Bad URL escaped!! ${badURLs}`);
        }
      }
    }

    // Remove duplicates from the urls collected:
    let urlsWithoutDuplicates = [...new Set(urls)];

    // Print scrapped urls.
    console.log(urlsWithoutDuplicates);
    // Wait 5 sec before closing the tab.
    // Once done close the current tab.
    fs.appendFileSync(
      `urls/yahoo-urls-${Date.now()}-${currentBrandName}.txt`, // generate a file with all the urls using the brand name.
      urlsWithoutDuplicates.join("\n")
    );

    /**
     * @end for brands
     */
  }

  await page.waitFor(5000);

  await page.close();

  await browser.close();
})();
