# syntax=docker/dockerfile:1
FROM python:3.9-bookworm AS base

# RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
# RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
#     --mount=type=cache,target=/var/lib/apt,sharing=locked \
#     export DEBIAN_FRONTEND=noninteractive \
#     && apt-get update \
#     && apt-get install -q -y --no-install-recommends \
#     curl git ffmpeg

RUN pip install -U pip setuptools wheel
RUN pip install pdm
RUN pip install --upgrade pip
COPY pyproject.toml pdm.lock /project/

WORKDIR /project
RUN pdm export -o /requirements.txt

#RUN grep -E 'git\+' /requirements.txt > /requirements-vcs.txt  # TODO added this to split hash and non hashes up
#RUN grep -E 'git\+' -v /requirements.txt > /requirements-hashed.txt

FROM python:3.9-bookworm AS prod

#COPY --from=base /requirements-vcs.txt /requirements-vcs.txt
#COPY --from=base /requirements-hashed.txt /requirements-hashed.txt
COPY --from=base /requirements.txt /requirements.txt

#RUN pip install cmake  # TODO added in cmake as was an issue that didn't let me install dm-tree

#RUN pip install --no-cache-dir  -r /requirements-vcs.txt
#RUN pip install --no-cache-dir -r /requirements-hashed.txt
RUN pip install --no-cache-dir -r /requirements.txt

RUN groupadd --gid 1000 app && useradd --uid 1000 --gid 1000 -m app

USER app
