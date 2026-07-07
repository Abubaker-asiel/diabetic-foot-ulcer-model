# Force Python 3.11 and correct architecture
FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

# Use 'printf' instead of 'echo' to correctly create newlines!
RUN printf "tensorflow\npillow\nstreamlit\n" > requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app code and model into the container
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
