# Use official Python image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose Streamlit port (if you're using Streamlit for frontend)
EXPOSE 8501

# Run the app (replace with your actual entrypoint)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
