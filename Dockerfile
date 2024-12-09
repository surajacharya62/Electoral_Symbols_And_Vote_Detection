FROM python:3.8-slim-buster

RUN apt update -y && \    
    apt install git awscli -y 


#Setting the working directory
WORKDIR /app

#Copying application files to the container
COPY . /app

#Installing Python dependencies
RUN pip install -r requirements.txt

CMD ["python3", "server.py"]