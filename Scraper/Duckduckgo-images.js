const puppeteer = require("puppeteer");
const fs = require("fs");
var glob = require("glob");

brands = glob.sync("keywords/*.txt");

(async () => {
  const browser = await puppeteer.launch({ headless: false });
  const page = await browser.newPage();
  await page.setViewport({ width: 1000, height: 900 });

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
      await page.goto("https://duckduckgo.com/", {
        waitUntil: "load"
      });

      // Wait for the search field to show up.
      await page.waitForSelector("input[name='q']");

      // Type the keyword value into the search field at a speed of 300ms between each keystroke
      await page.type("input[name='q']", keyWords[i], { delay: 300 });

      // Submit the seach form
      await page.click("input[type='submit']", { delay: 150 });
      
      await page.waitFor(5000);

      // Wait for the image field to show up.
      await page.waitForSelector("ul#duckbar_static > li.zcm__item > a.zcm__link.js-zci-link.js-zci-link--images");

      // Submit the image form
      await page.click("ul#duckbar_static > li.zcm__item > a.zcm__link.js-zci-link.js-zci-link--images", { delay: 150 });
      // --------------------------------------
      // Wait for search results to show up.
      await page.waitForSelector("div.tile.tile--img.has-detail");

      // Scrap urls from images.
      await page.waitForSelector(
        "#zci-images > div.js-tileview.tileview--grid > div.tile-wrap > div.zci__main.zci__main--tiles.js-tiles.has-nav.tileview__images.has-tiles--grid > div.tile.tile--img.has-detail"
      );

      await page.click(
        "#zci-images > div.js-tileview.tileview--grid > div.tile-wrap > div.zci__main.zci__main--tiles.js-tiles.has-nav.tileview__images.has-tiles--grid > div.tile.tile--img.has-detail",
        { delay: 150 }
      );
      
      let badURLs = 0;

      for (let j = 0; j < 200; j++) {
        let image_url = await page.evaluate(() => {
          if (
            document.querySelector(
              "#zci-images > div.detail.detail--slider.detail--images.detail--xd > div.detail__wrap > div.detail__panes.js-detail-panes > div:nth-child(1) > div.detail__inner > div.detail__media.detail__media--images > div.detail__media__img-wrapper.js-image-detail-wrapper > a.detail__media__img-link.js-detail-img.js-image-detail-link > img.detail__media__img-highres.js-detail-img.js-detail-img-high"
            )
          ) {
            return document.querySelector(
              "#zci-images > div.detail.detail--slider.detail--images.detail--xd > div.detail__wrap > div.detail__panes.js-detail-panes > div:nth-child(1) > div.detail__inner > div.detail__media.detail__media--images > div.detail__media__img-wrapper.js-image-detail-wrapper > a.detail__media__img-link.js-detail-img.js-image-detail-link > img.detail__media__img-highres.js-detail-img.js-detail-img-high"
            ).src;
          } else {
            return null;
          }
        });

        await page.waitForSelector(
          "#zci-images > div.detail.detail--slider.detail--images.detail--xd > div.detail__wrap > div.detail__controls > i.tile-nav--sm.tile-nav--sm--next.js-detail-next.can-scroll"
        );

        try {
          await page.click(
            "#zci-images > div.detail.detail--slider.detail--images.detail--xd > div.detail__wrap > div.detail__controls > i.tile-nav--sm.tile-nav--sm--next.js-detail-next.can-scroll",
            { delay: 150 }
          );
        } catch (error) {
          await page.click(
            "#zci-images > div.detail.detail--slider.detail--images.detail--xd > div.detail__wrap > div.detail__controls > i.tile-nav--sm.tile-nav--sm--next.js-detail-next.can-scroll",
            { delay: 150 }
          );
        }

        await page.waitFor(250);

        await page.waitForSelector(
          "#zci-images > div.detail.detail--slider.detail--images.detail--xd > div.detail__wrap > div.detail__panes.js-detail-panes > div:nth-child(1) > div.detail__inner > div.detail__media.detail__media--images > div.detail__media__img-wrapper.js-image-detail-wrapper > a.detail__media__img-link.js-detail-img.js-image-detail-link > img.detail__media__img-highres.js-detail-img.js-detail-img-high"
        );

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
      `urls/Duckduckgo-urls-${Date.now()}-${currentBrandName}.txt`, // generate a file with all the urls using the brand name.
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
