FROM cnbcool/default-dev-env:latest

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates curl gnupg; \
    mkdir -p /etc/apt/keyrings; \
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
      | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg; \
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
      | gpg --dearmor -o /etc/apt/keyrings/githubcli-archive-keyring.gpg; \
    chmod go+r /etc/apt/keyrings/nodesource.gpg /etc/apt/keyrings/githubcli-archive-keyring.gpg; \
    ARCH="$(dpkg --print-architecture)"; \
    echo "deb [arch=${ARCH} signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
      > /etc/apt/sources.list.d/github-cli.list; \
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_22.x nodistro main" \
      > /etc/apt/sources.list.d/nodesource.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      nodejs \
      gh \
      nano \
      htop \
      tmux \
      tree \
      ripgrep \
      fd-find \
      less; \
    npm install -g @openai/codex@latest; \
    npm install -g @github/copilot; \
    node --version; \
    npm --version; \
    gh --version; \
    codex --version; \
    rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    CODE_SERVER_BIN="$(command -v code-server || true)"; \
    if [ -z "$CODE_SERVER_BIN" ] && [ -x /home/ide/.local/bin/code-server ]; then CODE_SERVER_BIN="/home/ide/.local/bin/code-server"; fi; \
    if [ -z "$CODE_SERVER_BIN" ]; then echo "code-server binary not found"; exit 1; fi; \
    for ext in \
      ms-python.python \
      charliermarsh.ruff \
      redhat.vscode-yaml \
      EditorConfig.EditorConfig \
      MS-CEINTL.vscode-language-pack-zh-hans \
      eamodio.gitlens \
      yzhang.markdown-all-in-one \
      streetsidesoftware.code-spell-checker; do \
      "$CODE_SERVER_BIN" --install-extension "$ext" || echo "skip unavailable extension: $ext"; \
    done

ENV LANG=C.UTF-8
ENV LANGUAGE=C.UTF-8