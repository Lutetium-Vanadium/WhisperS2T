#---
# name: whisper_s2t
# group: speech
# depends: [whisper, tensorrt_llm, ctranslate2, huggingface_hub]
# requires: '>=34.1.0'
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

VOLUME /model-cache
WORKDIR /opt

COPY . /opt/whisper_s2t
WORKDIR /opt/whisper_s2t

# clear requirements, should all already be installed
RUN echo "" > requirements.txt
RUN pip install -e .

CMD bash
