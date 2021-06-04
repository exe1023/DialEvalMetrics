FROM cgsdfc/adem-1-master:latest
MAINTAINER cgsdfc <cgsdfc@126.com>

# Dependency
RUN pip3 install Flask

# Work around Click and Python3, see https://click.palletsprojects.com/en/7.x/python3/
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

EXPOSE 80

COPY . /root/adem
WORKDIR /root/adem

ENV FLASK_APP=server.py
ENV FLASK_ENV=development
CMD flask run -p 80 -h 0.0.0.0
