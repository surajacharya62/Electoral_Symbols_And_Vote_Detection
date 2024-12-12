FROM python:3.8-slim-buster

RUN apt update -y && \    
    apt install git awscli libgl1-mesa-glx -y 


#Setting the working directory
WORKDIR /app

#Copying application files to the container
COPY . /app

#Installing Python dependencies
RUN pip install -r requirements.txt

EXPOSE 8001
# CMD ["python3", "server.py"]
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]