import requests
import xml.etree.ElementTree as ET

SITEMAP_INDEX = "https://www.npr.org/sitemaps/all.xml"
OUTPUT_FILE = "npr_transcript_urls.txt"

def fetch_sitemap(url):
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return resp.text

def extract_transcript_urls(xml_text):
    root = ET.fromstring(xml_text)
    urls = []

    for elem in root.iter():
        if elem.tag.endswith("loc"):
            url = elem.text.strip()
            if "/transcripts/" in url:
                urls.append(url)

    return urls

def main():
    print("Fetching NPR sitemap index...")
    index_xml = fetch_sitemap(SITEMAP_INDEX)

    root = ET.fromstring(index_xml)
    sitemap_urls = [
        elem.text for elem in root.iter()
        if elem.tag.endswith("loc")
    ]

    print(f"Found {len(sitemap_urls)} sitemap files")

    all_transcript_urls = set()

    for i, sitemap_url in enumerate(sitemap_urls):
        try:
            print(f"[{i+1}/{len(sitemap_urls)}] Fetching {sitemap_url}")
            sitemap_xml = fetch_sitemap(sitemap_url)
            urls = extract_transcript_urls(sitemap_xml)
            all_transcript_urls.update(urls)
        except Exception as e:
            print(f"Failed to fetch {sitemap_url}: {e}")

    print(f"\nTotal transcript URLs found: {len(all_transcript_urls)}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for url in sorted(all_transcript_urls):
            f.write(url + "\n")

    print(f"Saved URLs to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
