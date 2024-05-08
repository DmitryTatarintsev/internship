@echo off
title Git Clone and Run Uvicorn. Github.com DmitryTatarintsev/internship/siz_detection

:: Клонируем репозиторий
cd %~dp0
git clone https://github.com/DmitryTatarintsev/internship.git

:: Переходим в папку internship/siz_detection
cd internship/siz_detection

:: Запускаем приложение uvicorn
python -m uvicorn main:app
