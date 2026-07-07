# Force Python 3.11
FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

# Install packages DIRECTLY, no requirements.txt needed
RUN pip install --no-cache-dir tensorflow pillow streamlit

# Copy your app code and model into the container
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
