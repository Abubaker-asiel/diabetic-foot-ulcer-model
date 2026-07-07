# Force Python 3.11 - Streamlit cannot override this
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Create a requirements file inside the container
RUN echo "tensorflow\npillow\nstreamlit" > requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app code and model into the container
COPY . .

# Expose the port Streamlit needs
EXPOSE 8501

# Command to run your app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
