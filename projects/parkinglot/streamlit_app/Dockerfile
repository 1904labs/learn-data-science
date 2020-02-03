FROM python:3.6

RUN apt-get update
#RUN apt-get install libgthread2.0-dev

RUN mkdir mrcnn
COPY ./Mask_RCNN/mrcnn /mrcnn
COPY ./Mask_RCNN/requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY /download_model.py download_model.py
RUN python download_model.py 
COPY ./images/* ./images/

COPY /st_parkinglot.py st_parkinglot.py
COPY /utils.py utils.py


EXPOSE 8501

CMD streamlit run st_parkinglot.py --server.enableCORS=False