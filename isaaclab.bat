@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Copyright (c) 2022-2025, The Isaac Lab Project Developers.
rem All rights reserved.
rem
rem SPDX-License-Identifier: BSD-3-Clause

rem Configurations
set "ISAACLAB_PATH=%~dp0"
goto main

rem Helper functions

rem extract Isaac Sim directory
:extract_isaacsim_path
rem Use the sym-link path to Isaac Sim directory
set isaac_path=%ISAACLAB_PATH%\_isaac_sim
rem Check if directory exists
if not exist "%isaac_path%" (
    rem Find the Python executable
    call :extract_python_exe
    rem retrieve the isaacsim path from the installed package
    set "isaac_path="
    for /f "delims=" %%i in ('!python_exe! -c "import isaacsim; import os; print(os.environ['ISAAC_PATH'])"') do (
        if not defined isaac_path (
            set "isaac_path=%%i"
        )
    )
)
rem Check if the directory exists
if not exist "%isaac_path%" (
    echo [错误] 无法找到Isaac Sim目录: %isaac_path%
    echo %tab%这可能是由于以下原因:
    echo %tab%1. 没有激活包含Isaac Sim pip包的Conda环境。
    echo %tab%2. Isaac Sim目录在默认路径不可用: %ISAACLAB_PATH%\_isaac_sim
    exit /b 1
)
goto :eof

rem extract the python from isaacsim
:extract_python_exe
rem check if using conda
if not "%CONDA_PREFIX%"=="" (
    rem use conda python
    set python_exe=%CONDA_PREFIX%\python.exe
) else (
    rem use kit python
    set python_exe=%ISAACLAB_PATH%\_isaac_sim\python.bat
)
rem check for if isaac sim was installed to system python
if not exist "%python_exe%" (
    set "python_exe="
    python -m pip show isaacsim-rl > nul 2>&1
    if %ERRORLEVEL% equ 0 (
        for /f "delims=" %%i in ('where python') do (
            if not defined python_exe (
                set "python_exe=%%i"
            )
        )
    )
)
if not exist "%python_exe%" (
    echo [错误] 无法在以下路径找到任何Python可执行文件: %python_exe%
    echo %tab%这可能是由于以下原因:
    echo %tab%1. Conda环境未激活。
    echo %tab%2. Python可执行文件在默认路径不可用: %ISAACLAB_PATH%\_isaac_sim\python.bat
    exit /b 1
)
goto :eof


rem extract the simulator exe from isaacsim
:extract_isaacsim_exe
call :extract_python_exe
call !python_exe! -m pip show isaacsim-rl > nul 2>&1
if errorlevel 1 (
    rem obtain isaacsim path
    call :extract_isaacsim_path
    rem python executable to use
    set isaacsim_exe=!isaac_path!\isaac-sim.bat
) else (
    rem if isaac sim installed from pip
    set isaacsim_exe=isaacsim isaacsim.exp.full
)
rem check if there is a python path available
if not exist "%isaacsim_exe%" (
    echo [错误] 在以下路径找不到isaac-sim可执行文件: %isaacsim_exe%
    exit /b 1
)
goto :eof


rem check if input directory is a python extension and install the module
:install_isaaclab_extension
echo %ext_folder%
rem retrieve the python executable
call :extract_python_exe
rem if the directory contains setup.py then install the python module
if exist "%ext_folder%\setup.py" (
    echo     模块: %ext_folder%
    call !python_exe! -m pip install --editable %ext_folder%
)
goto :eof


rem setup anaconda environment for Isaac Lab
:setup_conda_env
rem get environment name from input
set env_name=%conda_env_name%
rem check if conda is installed
where conda >nul 2>nul
if errorlevel 1 (
    echo [错误] 找不到Conda。请安装conda并重试。
    exit /b 1
)
rem check if the environment exists
call conda env list | findstr /c:"%env_name%" >nul
if %errorlevel% equ 0 (
    echo [信息] 名为'%env_name%'的Conda环境已存在。
) else (
    echo [信息] 创建名为'%env_name%'的conda环境...
    call conda create -y --name %env_name% python=3.10
)
rem cache current paths for later
set "cache_pythonpath=%PYTHONPATH%"
set "cache_ld_library_path=%LD_LIBRARY_PATH%"
rem clear any existing files
echo %CONDA_PREFIX%
del "%CONDA_PREFIX%\etc\conda\activate.d\setenv.bat" 2>nul
del "%CONDA_PREFIX%\etc\conda\deactivate.d\unsetenv.bat" 2>nul
rem activate the environment
call conda activate %env_name%
rem setup directories to load isaac-sim variables
mkdir "%CONDA_PREFIX%\etc\conda\activate.d" 2>nul
mkdir "%CONDA_PREFIX%\etc\conda\deactivate.d" 2>nul

rem obtain isaacsim path
call :extract_isaacsim_path
if exist "%isaac_path%" (
    rem add variables to environment during activation
    (
        echo @echo off
        echo rem for isaac-sim
        echo set "RESOURCE_NAME=IsaacSim"
        echo set CARB_APP_PATH=!isaac_path!\kit
        echo set EXP_PATH=!isaac_path!\apps
        echo set ISAAC_PATH=!isaac_path!
        echo set PYTHONPATH=%PYTHONPATH%;!isaac_path!\site
        echo.
        echo rem for isaac-lab
        echo doskey isaaclab=isaaclab.bat $*
    ) > "%CONDA_PREFIX%\etc\conda\activate.d\env_vars.bat"
    (
        echo $env:CARB_APP_PATH="!isaac_path!\kit"
        echo $env:EXP_PATH="!isaac_path!\apps"
        echo $env:ISAAC_PATH="!isaac_path!"
        echo $env:PYTHONPATH="%PYTHONPATH%;!isaac_path!\site"
        echo $env:RESOURCE_NAME="IsaacSim"
    ) > "%CONDA_PREFIX%\etc\conda\activate.d\env_vars.ps1"
) else (
    rem assume isaac sim will be installed from pip
    rem add variables to environment during activation
    (
        echo @echo off
        echo rem for isaac-sim
        echo set "RESOURCE_NAME=IsaacSim"
        echo.
        echo rem for isaac-lab
        echo doskey isaaclab=isaaclab.bat $*
    ) > "%CONDA_PREFIX%\etc\conda\activate.d\env_vars.bat"
    (
        echo $env:RESOURCE_NAME="IsaacSim"
    ) > "%CONDA_PREFIX%\etc\conda\activate.d\env_vars.ps1"
)

rem reactivate the environment to load the variables
call conda activate %env_name%

rem remove variables from environment during deactivation
(
    echo @echo off
    echo rem for isaac-sim
    echo set "CARB_APP_PATH="
    echo set "EXP_PATH="
    echo set "ISAAC_PATH="
    echo set "RESOURCE_NAME="
    echo.
    echo rem for isaac-lab
    echo doskey isaaclab =
    echo.
    echo rem restore paths
    echo set "PYTHONPATH=%cache_pythonpath%"
    echo set "LD_LIBRARY_PATH=%cache_ld_library_path%"
) > "%CONDA_PREFIX%\etc\conda\deactivate.d\unsetenv_vars.bat"
(
    echo $env:RESOURCE_NAME=""
    echo $env:PYTHONPATH="%cache_pythonpath%"
    echo $env:LD_LIBRARY_PATH="%cache_pythonpath%"
) > "%CONDA_PREFIX%\etc\conda\deactivate.d\unsetenv_vars.ps1"

rem install some extra dependencies
echo [信息] 安装额外依赖项(这可能需要几分钟)...
call conda install -c conda-forge -y importlib_metadata >nul 2>&1

rem deactivate the environment
call conda deactivate
rem add information to the user about alias
echo [信息] 为'isaaclab.bat'脚本向conda环境添加了'isaaclab'别名。
echo [信息] 创建了名为'%env_name%'的conda环境。
echo.
echo       1. 要激活环境，运行:                conda activate %env_name%
echo       2. 要安装Isaac Lab扩展，运行:      isaaclab -i
echo       3. 要执行格式化，运行:              isaaclab -f
echo       4. 要停用环境，运行:                conda deactivate
echo.
goto :eof


rem Update the vscode settings from template and Isaac Sim settings
:update_vscode_settings
echo [信息] 设置vscode设置...
rem Retrieve the python executable
call :extract_python_exe
rem Path to setup_vscode.py
set "setup_vscode_script=%ISAACLAB_PATH%\.vscode\tools\setup_vscode.py"
rem Check if the file exists before attempting to run it
if exist "%setup_vscode_script%" (
    call !python_exe! "%setup_vscode_script%"
) else (
    echo [警告] 找不到setup_vscode.py。中止vscode设置配置。
)
goto :eof


rem Print the usage description
:print_help
echo.
echo 用法: %~nx0 [-h] [-i] [-f] [-p] [-s] [-v] [-d] [-n] [-c] -- Isaac Lab管理工具。
echo.
echo 可选参数:
echo     -h, --help           显示帮助内容。
echo     -i, --install [LIB]  在Isaac Lab内安装扩展和学习框架作为额外依赖项。默认为'all'。
echo     -f, --format         运行pre-commit格式化代码并检查lint。
echo     -p, --python         运行由Isaac Sim提供的python可执行文件(python.bat)。
echo     -s, --sim            运行由Isaac Sim提供的模拟器可执行文件(isaac-sim.bat)。
echo     -t, --test           运行所有python单元测试。
echo     -v, --vscode         从模板生成VSCode设置文件。
echo     -d, --docs           使用sphinx从源代码构建文档。
echo     -n, --new            从模板创建新的外部项目或内部任务。
echo     -c, --conda [NAME]   为Isaac Lab创建conda环境。默认名称为'env_isaaclab'。
echo.
goto :eof


rem Main
:main

rem check argument provided
if "%~1"=="" (
    echo [错误] 未提供参数。
    call :print_help
    exit /b 1
)

rem pass the arguments
:loop
if "%~1"=="" goto :end
set "arg=%~1"

rem read the key
if "%arg%"=="-i" (
    rem install the python packages in isaaclab/source directory
    echo [信息] 在Isaac Lab仓库内安装扩展...
    call :extract_python_exe
    for /d %%d in ("%ISAACLAB_PATH%\source\*") do (
        set ext_folder="%%d"
        call :install_isaaclab_extension
    )
    rem install the python packages for supported reinforcement learning frameworks
    echo [信息] 安装额外要求，如学习框架...
    if "%~2"=="" (
        echo [信息] 安装所有rl框架...
        set framework_name=all
    ) else if "%~2"=="none" (
        echo [信息] 不会安装rl框架。
        set framework_name=none
        shift
    ) else (
        echo [信息] 安装rl框架: %2
        set framework_name=%2
        shift
    )
    rem install the rl-frameworks specified
    call !python_exe! -m pip install -e %ISAACLAB_PATH%\source\isaaclab_rl[!framework_name!]
    shift
) else if "%arg%"=="--install" (
    rem install the python packages in source directory
    echo [信息] 在Isaac Lab仓库内安装扩展...
    call :extract_python_exe
    for /d %%d in ("%ISAACLAB_PATH%\source\*") do (
        set ext_folder="%%d"
        call :install_isaaclab_extension
    )
    rem install the python packages for supported reinforcement learning frameworks
    echo [信息] 安装额外要求，如学习框架...
    if "%~2"=="" (
        echo [信息] 安装所有rl框架...
        set framework_name=all
    ) else if "%~2"=="none" (
        echo [信息] 不会安装rl框架。
        set framework_name=none
        shift
    ) else (
        echo [信息] 安装rl框架: %2
        set framework_name=%2
        shift
    )
    rem install the rl-frameworks specified
    call !python_exe! -m pip install -e %ISAACLAB_PATH%\source\isaaclab_rl[!framework_name!]
    rem update the vscode settings
    rem once we have a docker container, we need to disable vscode settings
    call :update_vscode_settings
    shift
) else if "%arg%"=="-c" (
    rem use default name if not provided
    if not "%~2"=="" (
        echo [信息] 使用conda环境名称: %2
        set conda_env_name=%2
        shift
    ) else (
        echo [信息] 使用默认conda环境名称: env_isaaclab
        set conda_env_name=env_isaaclab
    )
    call :setup_conda_env %conda_env_name%
    shift
) else if "%arg%"=="--conda" (
    rem use default name if not provided
    if not "%~2"=="" (
        echo [信息] 使用conda环境名称: %2
        set conda_env_name=%2
        shift
    ) else (
        echo [信息] 使用默认conda环境名称: env_isaaclab
        set conda_env_name=env_isaaclab
    )
    call :setup_conda_env %conda_env_name%
    shift
) else if "%arg%"=="-f" (
    rem reset the python path to avoid conflicts with pre-commit
    rem this is needed because the pre-commit hooks are installed in a separate virtual environment
    rem and it uses the system python to run the hooks
    if not "%CONDA_DEFAULT_ENV%"=="" (
        set cache_pythonpath=%PYTHONPATH%
        set PYTHONPATH=
    )

    rem run the formatter over the repository
    rem check if pre-commit is installed
    pip show pre-commit > nul 2>&1
    if errorlevel 1 (
        echo [信息] 安装pre-commit...
        pip install pre-commit
    )

    rem always execute inside the Isaac Lab directory
    echo [信息] 格式化仓库...
    pushd %ISAACLAB_PATH%
    call python -m pre_commit run --all-files
    popd >nul

    rem set the python path back to the original value
    if not "%CONDA_DEFAULT_ENV%"=="" (
        set PYTHONPATH=%cache_pythonpath%
    )
    goto :end
) else if "%arg%"=="--format" (
    rem reset the python path to avoid conflicts with pre-commit
    rem this is needed because the pre-commit hooks are installed in a separate virtual environment
    rem and it uses the system python to run the hooks
    if not "%CONDA_DEFAULT_ENV%"=="" (
        set cache_pythonpath=%PYTHONPATH%
        set PYTHONPATH=
    )

    rem run the formatter over the repository
    rem check if pre-commit is installed
    pip show pre-commit > nul 2>&1
    if errorlevel 1 (
        echo [信息] 安装pre-commit...
        pip install pre-commit
    )

    rem always execute inside the Isaac Lab directory
    echo [信息] 格式化仓库...
    pushd %ISAACLAB_PATH%
    call python -m pre_commit run --all-files
    popd >nul

    rem set the python path back to the original value
    if not "%CONDA_DEFAULT_ENV%"=="" (
        set PYTHONPATH=%cache_pythonpath%
    )
    goto :end
) else if "%arg%"=="-p" (
    rem run the python provided by Isaac Sim
    call :extract_python_exe
    echo [信息] 使用来自以下位置的python: !python_exe!
    REM Loop through all arguments - mimic shift
    set "allArgs="
    for %%a in (%*) do (
        REM Append each argument to the variable, skip the first one
        if defined skip (
            set "allArgs=!allArgs! %%a"
        ) else (
            set "skip=1"
        )
    )
    !python_exe! !allArgs!
    goto :end
) else if "%arg%"=="--python" (
    rem run the python provided by Isaac Sim
    call :extract_python_exe
    echo [信息] 使用来自以下位置的python: !python_exe!
    REM Loop through all arguments - mimic shift
    set "allArgs="
    for %%a in (%*) do (
        REM Append each argument to the variable, skip the first one
        if defined skip (
            set "allArgs=!allArgs! %%a"
        ) else (
            set "skip=1"
        )
    )
    !python_exe! !allArgs!
    goto :end
) else if "%arg%"=="-s" (
    rem run the simulator exe provided by isaacsim
    call :extract_isaacsim_exe
    echo [信息] 从以下位置运行isaac-sim: !isaacsim_exe!
    set "allArgs="
    for %%a in (%*) do (
        REM Append each argument to the variable, skip the first one
        if defined skip (
            set "allArgs=!allArgs! %%a"
        ) else (
            set "skip=1"
        )
    )
    !isaacsim_exe! --ext-folder %ISAACLAB_PATH%\source !allArgs1
    goto :end
) else if "%arg%"=="--sim" (
    rem run the simulator exe provided by Isaac Sim
    call :extract_isaacsim_exe
    echo [信息] 从以下位置运行isaac-sim: !isaacsim_exe!
    set "allArgs="
    for %%a in (%*) do (
        REM Append each argument to the variable, skip the first one
        if defined skip (
            set "allArgs=!allArgs! %%a"
        ) else (
            set "skip=1"
        )
    )
    !isaacsim_exe! --ext-folder %ISAACLAB_PATH%\source !allArgs1
    goto :end
) else if "%arg%"=="-n" (
    rem run the template generator script
    call :extract_python_exe
    set "allArgs="
    for %%a in (%*) do (
        REM Append each argument to the variable, skip the first one
        if defined skip (
            set "allArgs=!allArgs! %%a"
        ) else (
            set "skip=1"
        )
    )
    echo [信息] 安装模板依赖项...
    !python_exe! -m pip install -q -r tools\template\requirements.txt
    echo.
    echo [信息] 运行模板生成器...
    echo.
    !python_exe! tools\template\cli.py !allArgs!
    goto :end
) else if "%arg%"=="--new" (
    rem run the template generator script
    call :extract_python_exe
    set "allArgs="
    for %%a in (%*) do (
        REM Append each argument to the variable, skip the first one
        if defined skip (
            set "allArgs=!allArgs! %%a"
        ) else (
            set "skip=1"
        )
    )
    echo [信息] 安装模板依赖项...
    !python_exe! -m pip install -q -r tools\template\requirements.txt
    echo.
    echo [信息] 运行模板生成器...
    echo.
    !python_exe! tools\template\cli.py !allArgs!
    goto :end
) else if "%arg%"=="-t" (
    rem run the python provided by Isaac Sim
    call :extract_python_exe
    set "allArgs="
    for %%a in (%*) do (
        REM Append each argument to the variable, skip the first one
        if defined skip (
            set "allArgs=!allArgs! %%a"
        ) else (
            set "skip=1"
        )
    )
    !python_exe! tools\run_all_tests.py !allArgs!
    goto :end
) else if "%arg%"=="--test" (
    rem run the python provided by Isaac Sim
    call :extract_python_exe
    set "allArgs="
    for %%a in (%*) do (
        REM Append each argument to the variable, skip the first one
        if defined skip (
            set "allArgs=!allArgs! %%a"
        ) else (
            set "skip=1"
        )
    )
    !python_exe! tools\run_all_tests.py !allArgs!
    goto :end
) else if "%arg%"=="-v" (
    rem update the vscode settings
    call :update_vscode_settings
    shift
    goto :end
) else if "%arg%"=="--vscode" (
    rem update the vscode settings
    call :update_vscode_settings
    shift
    goto :end
) else if "%arg%"=="-d" (
    rem build the documentation
    echo [信息] 构建文档...
    call :extract_python_exe
    pushd %ISAACLAB_PATH%\docs
    call !python_exe! -m pip install -r requirements.txt >nul
    call !python_exe! -m sphinx -b html -d _build\doctrees . _build\html
    echo [信息] 要在默认浏览器中打开文档，运行:
    echo xdg-open "%ISAACLAB_PATH%\docs\_build\html\index.html"
    popd >nul
    shift
    goto :end
) else if "%arg%"=="--docs" (
    rem build the documentation
    echo [信息] 构建文档...
    call :extract_python_exe
    pushd %ISAACLAB_PATH%\docs
    call !python_exe! -m pip install -r requirements.txt >nul
    call !python_exe! -m sphinx -b html -d _build\doctrees . _build\current
    echo [信息] 要在默认浏览器中打开文档，运行:
    echo xdg-open "%ISAACLAB_PATH%\docs\_build\current\index.html"
    popd >nul
    shift
    goto :end
) else if "%arg%"=="-h" (
    call :print_help
    goto :end
) else if "%arg%"=="--help" (
    call :print_help
    goto :end
) else (
    echo 提供了无效参数: %arg%
    call :print_help
    exit /b 1
)
goto loop

:end
exit /b 0
