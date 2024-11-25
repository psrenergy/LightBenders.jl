@echo off

SET BASEPATH=%~dp0
SET REVISE_PATH="%BASEPATH%\revise"

CALL %JULIA_194% --project=%REVISE_PATH% --load=%REVISE_PATH%\revise_load_script.jl
