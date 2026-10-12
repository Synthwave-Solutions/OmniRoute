FROM synthpulse-omniroute-base AS pi
USER root
RUN apt-get update && apt-get install -y --no-install-recommends python3 git && rm -rf /var/lib/apt/lists/*
RUN npm install -g @openai/codex@0.162.1 @anthropic-ai/claude-code@2.1.296 && node /usr/local/lib/node_modules/@anthropic-ai/claude-code/install.cjs && claude --version && npm cache clean --force
COPY synthpulse/claude-bridge /opt/synthpulse/claude-bridge
ENV HERMES_AGENT_DIR=/opt/synthpulse/claude-bridge/hermes-support
LABEL org.opencontainers.image.source="https://github.com/Synthwave-Solutions/OmniRoute"
USER node
