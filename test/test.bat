@echo off

SET BASEPATH=%~dp0

%JULIA_1106% --project=%BASEPATH%\.. -e "import Pkg; Pkg.test()"