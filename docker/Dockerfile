FROM ubuntu:18.04
COPY ./mnist /exp1/mnist
COPY ./requirement.txt /exp1/requirement.txt
RUN apt-get update && apt-get install -y python3.7 python3-pip 
RUN pip3 install --no-cache-dir -r /exp1/requirement.txt
CMD ["python3" , "/exp1/mnist/API/api.py"]