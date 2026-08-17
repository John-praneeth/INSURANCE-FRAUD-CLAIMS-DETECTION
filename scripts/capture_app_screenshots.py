"""
Script to capture high-resolution screenshots of the Streamlit application pages.
"""

import os
import time
from playwright.sync_api import sync_playwright

output_dir = "reports/figures"
os.makedirs(output_dir, exist_ok=True)

def capture_screenshots():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 900}, device_scale_factor=2)
        page = context.new_page()

        print("Navigating to Streamlit app at http://localhost:8501...")
        page.goto("http://localhost:8501", wait_until="networkidle")
        time.sleep(3)

        # 1. Capture Dashboard Page
        print("Capturing 1: Dashboard Page...")
        page.screenshot(path=os.path.join(output_dir, "screenshot_dashboard_page.png"), full_page=False)

        # 2. Capture Claim Risk Screening Page
        print("Capturing 2: Claim Risk Screening Form...")
        page.locator("text=Claim Risk Screening").click()
        time.sleep(2)
        page.screenshot(path=os.path.join(output_dir, "screenshot_claim_screening_form.png"), full_page=False)

        # Submit the form to show risk analysis output
        print("Submitting claim analysis form...")
        page.locator("button:has-text('Analyze Claim Risk')").click()
        time.sleep(2)
        page.screenshot(path=os.path.join(output_dir, "screenshot_claim_screening_result.png"), full_page=False)

        # 3. Capture Model Analytics Page
        print("Capturing 3: Model Analytics Page...")
        page.locator("text=Model Analytics").click()
        time.sleep(2)
        page.screenshot(path=os.path.join(output_dir, "screenshot_model_analytics_page.png"), full_page=False)

        # 4. Capture About System Page
        print("Capturing 4: About System Page...")
        page.locator("text=About System").click()
        time.sleep(2)
        page.screenshot(path=os.path.join(output_dir, "screenshot_about_system_page.png"), full_page=False)

        browser.close()
        print("All screenshots successfully captured in reports/figures/!")

if __name__ == "__main__":
    capture_screenshots()
