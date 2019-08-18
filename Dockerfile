FROM jupyter/minimal-notebook:latest

COPY --chown=1000:100 src /home/$NB_USER/src

COPY requirements.txt /home/$NB_USER/src/requirements.txt

COPY preproc.args /home/$NB_USER/preproc.args

RUN pip install --default-timeout=60 -r /home/$NB_USER/src/requirements.txt

WORKDIR /home/$NB_USER/

#EXPOSE 80

