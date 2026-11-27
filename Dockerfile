FROM python:3.12-slim

WORKDIR /srv/app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements-app.txt ./
RUN pip install --no-cache-dir -r requirements-app.txt

COPY app ./app

ENV PORT=7860
EXPOSE 7860

CMD ["python", "-m", "app.server", "--host", "0.0.0.0", "--port", "7860"]
