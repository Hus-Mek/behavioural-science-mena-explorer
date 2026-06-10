# Behavioural Science Middle East Scraper & Analyser

This project provides tools to scrape academic papers (primarily from arXiv) related to behavioural science in the Middle East/Arab/Saudi context, and perform basic analysis on the collected data.

## Directory Structure
```
localled/
├── scraper.py          # Main scraper for arXiv
├── analyser.py         # Analysis module
├── requirements.txt    # Python dependencies
├── data/
│   ├── raw/            # Scraped JSON data and PDFs
│   ├── processed/      # Processed data (if any)
│   └── results/        # Analysis reports
└── README.md           # This file
```

## Installation
```bash
pip install -r requirements.txt
```

## Usage

### 1. Scrape Papers
Run the scraper to collect papers from arXiv:
```bash
python scraper.py
```
This will:
- Search for papers with "Middle East" AND "behavioural science" in all fields
- Download metadata (title, authors, abstract, etc.)
- Save raw JSON data to `data/raw/`
- Optionally download PDFs (uncomment the download line in scraper.py)

### 2. Analyse Data
After scraping, run the analyser:
```bash
python analyser.py
```
This will:
- Load the most recent scraped data
- Compute basic statistics (date range, author counts, yearly trends)
- Extract top keywords from titles and abstracts
- Count occurrences of behavioural science terms
- Generate a JSON report in `data/results/`
- Print a summary to console

## Customisation

### Adjust Search Query
In `scraper.py`, modify the `build_query` method to change the search terms.
Examples:
- Focus on specific countries: `all:"Saudi Arabia" AND all:"behavioural"`
- Different date ranges: adjust `sortBy` or add date filters to query
- Other sources: extend scraper to use other APIs (Semantic Scholar, Crossref, etc.)

### PDF Download
The scraper includes a `download_pdfs` method (currently commented out in `__main__`).
Uncomment to download PDFs for further text analysis.

### Analysis Enhancements
In `analyser.py`, you can:
- Add more sophisticated NLP (topic modeling, sentiment analysis)
- Integrate with local LLMs for summarisation
- Extract geographical entities (cities, countries) using NER
- Perform temporal trend analysis on specific topics

## Dependencies
- requests: for HTTP requests to arXiv API
- beautifulsoup4: for potential HTML parsing (if extending to other sites)
- PyPDF2: for PDF text extraction (if PDFs downloaded)
- numpy, pandas: for data handling
- scikit-learn: for potential machine learning analysis
- nltk, spaCy: for natural language processing

## Ethical Considerations
- Respect arXiv's API usage policy (the scraper includes a 3-second delay between requests)
- Only use collected data for research/educational purposes
- Credit original authors when sharing results
- Consider privacy and consent if analysing human subjects data within papers

## Example Output
After running both scripts, you might see output like:
```
Fetching results 0-100...
...
Saved 42 papers to data/raw/arxiv_me_behavioural_1717890123.json

=== BEHAVIOURAL SCIENCE MIDDLE EAST ANALYSIS ===
Total papers analyzed: 42
Date range: 2020-03-15T00:00:00+00:00 to 2024-02-28T00:00:00+00:00
Average authors per paper: 3.45
Papers per year: {2020: 5, 2021: 8, 2022: 10, 2023: 12, 2024: 7}

Top keywords in titles:
  behaviour: 12
  middle: 10
  east: 9
  ...

Top keywords in summaries:
  intervention: 18
  decision: 15
  nudge: 12
  ...

Behavioural science term frequencies:
  behaviour: 38
  decision: 32
  nudge: 25
  ...
```

## Future Extensions
- Add support for other databases (PubMed, IEEE Xplore, SpringerLink)
- Implement Arabic language detection and processing
- Build a searchable web interface for the collected papers
- Add citation network analysis
- Integrate with Zotero/Mendeley for reference management

## License
MIT