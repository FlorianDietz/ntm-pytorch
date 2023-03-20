FROM anibali/pytorch:1.10.2-cuda11.3-ubuntu20.04

RUN pip install GitPython
RUN pip install pandas
RUN pip install ipython
RUN pip install matplotlib
RUN pip install graphviz
RUN pip install tensorboard
RUN pip install setuptools==58.2.0
RUN pip install msgpack

RUN pip install numpy
RUN pip install tensorboard_logger
RUN pip install matplotlib
RUN pip install tqdm
RUN pip install Pillow
RUN pip install tensorflow