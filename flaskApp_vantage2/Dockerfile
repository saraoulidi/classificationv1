FROM python:3.6.11

WORKDIR /usr/src/app

COPY requirements.txt ./

EXPOSE 5000

RUN pip install -r requirements.txt

COPY . .

CMD ["python", "-m", "flask", "run"]