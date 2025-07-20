# Use an official lightweight Python image and specify the platform
# as required by the hackathon [cite: 107, 108]
FROM --platform=linux/amd64 python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker's layer caching
COPY requirements.txt .

# Install the Python dependencies
# The --no-cache-dir option keeps the image smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your project files into the container's working directory
# This includes your Python scripts and the trained model (.pkl) files
COPY . .

# The command that will be executed when the container starts
# This runs your main script, which processes files from /app/input to /app/output
CMD ["python", "main.py"]