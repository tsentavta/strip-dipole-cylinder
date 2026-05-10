# =============================================================================
# strip-dipole-cylinder — Dockerfile
# =============================================================================
# Контейнер для запуска веб-приложения strip-dipole-cylinder.
#
# Использование:
#   docker build -t strip-dipole-cylinder .
#   docker run -p 8501:8501 strip-dipole-cylinder
#
# Затем откройте http://localhost:8501 в браузере.
#
# Multi-stage build для минимизации финального образа.
# =============================================================================

# Стадия 1: builder — установка зависимостей в виртуальное окружение
FROM python:3.11-slim AS builder

LABEL stage=builder

# Устанавливаем системные зависимости для сборки
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Создаём виртуальное окружение
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Устанавливаем зависимости
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Стадия 2: runtime — финальный образ
# =============================================================================
FROM python:3.11-slim AS runtime

LABEL maintainer="Кузнецов Е.М. <kuznetsov@example.com>"
LABEL description="Программный комплекс для расчёта характеристик полоскового вибраторного излучателя на диэлектрическом цилиндре"
LABEL version="1.0.0"

# Системные зависимости для runtime (минимум для NumPy/SciPy/Matplotlib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Копируем виртуальное окружение из builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Не запускаем под root — создаём непривилегированного пользователя
RUN groupadd --system --gid 1001 streamlit && \
    useradd --system --uid 1001 --gid streamlit --create-home --shell /bin/bash streamlit

WORKDIR /app

# Копируем код приложения
COPY --chown=streamlit:streamlit strip_dipole_cylinder/ ./strip_dipole_cylinder/
COPY --chown=streamlit:streamlit app.py ./
COPY --chown=streamlit:streamlit .streamlit/ ./.streamlit/
COPY --chown=streamlit:streamlit tests/ ./tests/

USER streamlit

# Настройки Streamlit для работы в контейнере
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8501

# Health check для оркестраторов (Docker Swarm, Kubernetes)
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py"]
