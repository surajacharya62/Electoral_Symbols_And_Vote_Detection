FROM python:3.8-slim-buster

# Update and install necessary libraries
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    git \
    awscli \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 && \
    apt-get clean

# Setting the working directory
WORKDIR /app

# Copying application files to the container
COPY . /app

# Installing Python dependencies
RUN pip install -r requirements.txt

# Expose port 8001 for the app
EXPOSE 8001

# Run the app with Uvicorn
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
