FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code. config.json is deliberately NOT copied: it is
# gitignored (holds API keys), so it does not exist in a CI checkout and the
# COPY made every CI docker build fail. Keys come from the environment
# (OPENROUTER_API_KEY / GEMINI_API_KEY) or a volume-mounted config.json.
COPY server.py .
COPY app/ app/
COPY scraper.py .
COPY enrichment.py .
COPY build_paper_graph.py .
COPY prompts.json .
COPY grey_sources/ grey_sources/
COPY data/ data/
COPY index.html .

# Create directories for runtime data
RUN mkdir -p /app/data/raw /app/data/analyses /app/results

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:3000/api/analysis')" || exit 1

# Run server
CMD ["python", "server.py", "3000"]
