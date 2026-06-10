FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY server.py .
COPY scraper.py .
COPY prompts.json .
COPY config.json .
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
