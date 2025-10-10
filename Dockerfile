# Usa a imagem base oficial do Python 3.10
FROM python:3.10-slim

# Define a variável de ambiente para que os logs sejam exibidos imediatamente
ENV PYTHONUNBUFFERED 1

# Define o diretório de trabalho dentro do contêiner
WORKDIR /app

# Copia os arquivos de dependência e o modelo primeiro para aproveitar o cache do Docker
COPY requirements.txt .
COPY ia_wdo_v1_ticks.h5 .

# Instala todas as dependências do Python
# O pip resolve a versão correta do TensorFlow para o Python 3.10
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código principal (seu arquivo main.py)
# Assumimos que você moveu o código para main.py na raiz
COPY main.py .

# Exponha a porta que o servidor Gunicorn usará (padrão do Fly.io)
EXPOSE 8080

# Comando para iniciar o servidor Gunicorn, apontando para sua aplicação Flask
# O Gunicorn é essencial para rodar o Flask em produção.
# 'main' é o nome do arquivo Python (main.py)
# 'app' é a instância do Flask dentro do main.py (app = Flask(__name__))
CMD exec gunicorn --bind :8080 --workers 4 --threads 2 main:app
