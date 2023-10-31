FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ENV PROJ_DIR /BLASComp
RUN mkdir ${PROJ_DIR}
RUN apt-get update
RUN apt-get install cmake -y
WORKDIR ${PROJ_DIR}
ENTRYPOINT ["bash"]
