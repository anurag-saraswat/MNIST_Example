FROM ubuntu:18.04
COPY ./mnist /exp1/mnist
COPY ./requirement.txt /exp1/requirement.txt
RUN apt-get update && apt-get install -y python3.7 python3-pip 
RUN pip3 install --no-cache-dir -r /exp1/requirement.txt
WORKDIR /exp1
CMD ["python3" , "./mnist/Final_exam.py"]