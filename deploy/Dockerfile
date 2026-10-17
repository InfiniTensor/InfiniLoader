FROM python:3.12-slim
WORKDIR /opt/infiniloader
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir . \
    && mkdir -p /var/cache/infiniloader \
    && chown 10001:10001 /var/cache/infiniloader
ENV ILOADER_BASE_DIR=/var/cache/infiniloader
USER 10001:10001
ENTRYPOINT ["iloader"]
CMD ["doctor", "--json"]
