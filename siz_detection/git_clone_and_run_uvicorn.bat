@echo off
title Git Clone and Run Uvicorn

:: Клонируем репозиторий
cd %~dp0
git clone https://github.com/DmitryTatarintsev/internship.git

:: Переходим в папку internship/siz_detection
cd internship/siz_detection

:: Запускаем приложение uvicorn
python -m uvicorn main:app

:: Открываем приложение в браузере
start http://localhost:8000/docs

:: Ожидаем завершение работы uvicorn (если он не был остановлен вручную)
tasklist /FI "IMAGENAME eq python.exe" | find /I "/FI" > nul || timeout /t 60

:: Выход из командной строки
exit
