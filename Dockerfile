FROM python:3.10-slim

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN git clone https://github.com/JSiddharth1105/mlopsss.git .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["streamlit", "run", "mlops.py", "--server.port=8080","--server.address=0.0.0.0","--server.enableCORS=false"]
