@echo off

SET BASEPATH=%~dp0
SET REVISE_PATH="%BASEPATH%\revise"

%JULIA_194% --color=yes --project=%REVISE_PATH% --load=%REVISE_PATH%\revise_load_script.jl
