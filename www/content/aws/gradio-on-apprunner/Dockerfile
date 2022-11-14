FROM python:3.9

WORKDIR /workspace
ENV HOME=/workspace
ADD requirements.txt /workspace/requirements.txt
RUN pip install -U pip &&\
  pip install --no-cache-dir -r /workspace/requirements.txt
ADD serve.py /workspace/serve.py

EXPOSE 8080

CMD [ "python3" , "serve.py" ]