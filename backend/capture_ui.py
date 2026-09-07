import asyncio
from playwright.async_api import async_playwright
import os
import shutil

async def capture_screenshots():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        # Navigate to the frontend
        await page.goto("http://localhost:5173/")
        
        # We need a place to save screenshots
        output_dir = "/Users/sukesh/.gemini/antigravity-ide/brain/79ee20dc-1496-4344-969c-bba8dbb932d1/scratch"
        os.makedirs(output_dir, exist_ok=True)
        
        test_cases = [
            ("test-case-a", "test_a_single_optical.png"),
            ("test-case-b", "test_b_two_optical.png"),
            ("test-case-c", "test_c_optical_sar.png"),
            ("test-case-d", "test_d_unsupported.png"),
            ("test-case-e", "test_e_no_georef.png"),
        ]
        
        for btn_id, filename in test_cases:
            print(f"Running {btn_id}...")
            
            # Click the test harness button
            await page.click(f"#{btn_id}")
            
            # Wait for processing to finish (either success or error panel appears)
            # We wait until the empty state or processing spinner is gone
            # Result panel shows either class .result-panel--error or just .result-panel without --processing
            await page.wait_for_selector(".result-panel:not(.result-panel--processing)", timeout=60000)
            
            # Wait a bit for animations/maps to load
            await page.wait_for_timeout(2000)
            
            filepath = os.path.join(output_dir, filename)
            await page.screenshot(path=filepath, full_page=True)
            print(f"Captured {filename}")
            
            # Since Test Harness actually triggers onLoadCase which immediately submits a new query
            # we just wait for it.
            
        await browser.close()
        print("All screenshots captured.")

if __name__ == "__main__":
    asyncio.run(capture_screenshots())
