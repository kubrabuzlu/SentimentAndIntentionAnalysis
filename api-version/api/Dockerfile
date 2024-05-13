FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

RUN mkdir -p /.cache && chmod -R a+w /.cache

COPY . .

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "7860"]