# Usa uma imagem base com Python 3.10 (compatível com TensorFlow)
FROM python:3.10-slim

# Define o diretório de trabalho
WORKDIR /app

# Copia e instala as dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o resto do projeto (código e o modelo .h5)
COPY . .

# Define a porta de execução do servidor
ENV PORT 8080
EXPOSE 8080

# Comando para iniciar a aplicação usando gunicorn
CMD exec gunicorn --bind :$PORT api.index:app
