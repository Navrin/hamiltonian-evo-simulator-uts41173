# part from https://stackoverflow.com/questions/65768775/how-do-i-integrate-pyenv-poetry-and-docker
FROM debian


RUN apt-get update
RUN apt-get install -y --no-install-recommends make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

ENV HOME="/root"
WORKDIR ${HOME}
RUN apt-get install -y git
RUN git clone --depth=1 https://github.com/pyenv/pyenv.git .pyenv
ENV PYENV_ROOT="${HOME}/.pyenv"
ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}:${HOME}/.local/bin"

ENV PYTHON_VERSION=3.11.6
RUN pyenv install ${PYTHON_VERSION}
RUN pyenv global ${PYTHON_VERSION}

RUN curl -sSL https://install.python-poetry.org | python -

WORKDIR /app

# RUN apt-get install -y mecab-ipadic-utf8
# RUN touch /usr/local/etc/mecabrc

COPY poetry.lock pyproject.toml ./

RUN /bin/bash -c 'cd /app && poetry install'

COPY . /app/

RUN /bin/bash -c 'cd /app && source $(poetry env info --path)/bin/activate && poetry install --compile'

ENTRYPOINT [ "/bin/bash", "-c", "source $(poetry env info --path)/bin/activate && python hamil_clever_sim/entry.py" ]
