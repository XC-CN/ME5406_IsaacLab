#!/usr/bin/env bash

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

#==
# 配置
#==

# 如果发生错误则退出
set -e

# 设置制表符空格
tabs 4

# 获取源目录
export ISAACLAB_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

#==
# 辅助函数
#==

# 检查是否在docker中运行
is_docker() {
    [ -f /.dockerenv ] || \
    grep -q docker /proc/1/cgroup || \
    [[ $(cat /proc/1/comm) == "containerd-shim" ]] || \
    grep -q docker /proc/mounts || \
    [[ "$(hostname)" == *"."* ]]
}

# 提取isaac sim路径
extract_isaacsim_path() {
    # 使用指向Isaac Sim目录的符号链接路径
    local isaac_path=${ISAACLAB_PATH}/_isaac_sim
    # 如果上述路径不可用，尝试使用python查找路径
    if [ ! -d "${isaac_path}" ]; then
        # 使用python可执行文件获取路径
        local python_exe=$(extract_python_exe)
        # 通过导入isaac sim并获取环境路径
        if [ $(${python_exe} -m pip list | grep -c 'isaacsim-rl') -gt 0 ]; then
            local isaac_path=$(${python_exe} -c "import isaacsim; import os; print(os.environ['ISAAC_PATH'])")
        fi
    fi
    # 检查是否有可用路径
    if [ ! -d "${isaac_path}" ]; then
        # 如果找不到路径则抛出错误
        echo -e "[错误] 无法找到Isaac Sim目录: '${isaac_path}'" >&2
        echo -e "\t这可能是由于以下原因:" >&2
        echo -e "\t1. Conda环境未激活。" >&2
        echo -e "\t2. Isaac Sim pip包 'isaacsim-rl' 未安装。" >&2
        echo -e "\t3. Isaac Sim目录在默认路径不可用: ${ISAACLAB_PATH}/_isaac_sim" >&2
        # 退出脚本
        exit 1
    fi
    # 返回结果
    echo ${isaac_path}
}

# 从isaacsim提取python
extract_python_exe() {
    # 检查是否使用conda
    if ! [[ -z "${CONDA_PREFIX}" ]]; then
        # 使用conda python
        local python_exe=${CONDA_PREFIX}/bin/python
    else
        # 使用工具包python
        local python_exe=${ISAACLAB_PATH}/_isaac_sim/python.sh

    if [ ! -f "${python_exe}" ]; then
            # 注意：我们需要检查系统python，例如在docker中
            # 在docker内部，如果用户安装到系统python，我们需要使用它
            # 否则，使用工具包中的python
            if [ $(python -m pip list | grep -c 'isaacsim-rl') -gt 0 ]; then
                local python_exe=$(which python)
            fi
        fi
    fi
    # 检查是否有可用的python路径
    if [ ! -f "${python_exe}" ]; then
        echo -e "[错误] 无法在路径找到任何Python可执行文件: '${python_exe}'" >&2
        echo -e "\t这可能是由于以下原因:" >&2
        echo -e "\t1. Conda环境未激活。" >&2
        echo -e "\t2. Isaac Sim pip包 'isaacsim-rl' 未安装。" >&2
        echo -e "\t3. Python可执行文件在默认路径不可用: ${ISAACLAB_PATH}/_isaac_sim/python.sh" >&2
        exit 1
    fi
    # 返回结果
    echo ${python_exe}
}

# 从isaacsim提取模拟器可执行文件
extract_isaacsim_exe() {
    # 获取isaac sim路径
    local isaac_path=$(extract_isaacsim_path)
    # 要使用的isaac sim可执行文件
    local isaacsim_exe=${isaac_path}/isaac-sim.sh
    # 检查是否有可用的python路径
    if [ ! -f "${isaacsim_exe}" ]; then
        # 检查使用Isaac Sim pip安装
        # 注意：通过pip安装的Isaac Sim只能来自直接的
        # python环境，所以我们可以直接在这里使用'python'
        if [ $(python -m pip list | grep -c 'isaacsim-rl') -gt 0 ]; then
            # Isaac Sim - Python包入口点
            local isaacsim_exe="isaacsim isaacsim.exp.full"
        else
            echo "[错误] 在路径找不到Isaac Sim可执行文件: ${isaac_path}" >&2
            exit 1
        fi
    fi
    # 返回结果
    echo ${isaacsim_exe}
}

# 检查输入目录是否为python扩展并安装模块
install_isaaclab_extension() {
    # 获取python可执行文件
    python_exe=$(extract_python_exe)
    # 如果目录包含setup.py，则安装python模块
    if [ -f "$1/setup.py" ]; then
        echo -e "\t 模块: $1"
        ${python_exe} -m pip install --editable $1
    fi
}

# 为Isaac Lab设置anaconda环境
setup_conda_env() {
    # 从输入获取环境名称
    local env_name=$1
    # 检查是否安装了conda
    if ! command -v conda &> /dev/null
    then
        echo "[错误] 找不到Conda。请安装conda并重试。"
        exit 1
    fi

    # 检查环境是否存在
    if { conda env list | grep -w ${env_name}; } >/dev/null 2>&1; then
        echo -e "[信息] 名为'${env_name}'的Conda环境已存在。"
    else
        echo -e "[信息] 创建名为'${env_name}'的conda环境..."
        conda create -y --name ${env_name} python=3.10
    fi

    # 缓存当前路径以便后续使用
    cache_pythonpath=$PYTHONPATH
    cache_ld_library_path=$LD_LIBRARY_PATH
    # 清除任何现有文件
    rm -f ${CONDA_PREFIX}/etc/conda/activate.d/setenv.sh
    rm -f ${CONDA_PREFIX}/etc/conda/deactivate.d/unsetenv.sh
    # 激活环境
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate ${env_name}
    # 设置目录以加载Isaac Sim变量
    mkdir -p ${CONDA_PREFIX}/etc/conda/activate.d
    mkdir -p ${CONDA_PREFIX}/etc/conda/deactivate.d

    # 在激活期间向环境添加变量
    printf '%s\n' '#!/usr/bin/env bash' '' \
        '# for Isaac Lab' \
        'export ISAACLAB_PATH='${ISAACLAB_PATH}'' \
        'alias isaaclab='${ISAACLAB_PATH}'/isaaclab.sh' \
        '' \
        '# 如果不是无头运行则显示图标' \
        'export RESOURCE_NAME="IsaacSim"' \
        '' > ${CONDA_PREFIX}/etc/conda/activate.d/setenv.sh

    # 检查是否有_isaac_sim目录 -> 如果有，表示已安装二进制文件。
    # 我们需要设置conda变量来加载二进制文件
    local isaacsim_setup_conda_env_script=${ISAACLAB_PATH}/_isaac_sim/setup_conda_env.sh

    if [ -f "${isaacsim_setup_conda_env_script}" ]; then
        # 在激活期间向环境添加变量
        printf '%s\n' \
            '# for Isaac Sim' \
            'source '${isaacsim_setup_conda_env_script}'' \
            '' >> ${CONDA_PREFIX}/etc/conda/activate.d/setenv.sh
    fi

    # 重新激活环境以加载变量
    # 需要这样做是因为deactivate会抱怨Isaac Lab别名，因为它否则不存在
    conda activate ${env_name}

    # 在停用期间从环境中删除变量
    printf '%s\n' '#!/usr/bin/env bash' '' \
        '# for Isaac Lab' \
        'unalias isaaclab &>/dev/null' \
        'unset ISAACLAB_PATH' \
        '' \
        '# 恢复路径' \
        'export PYTHONPATH='${cache_pythonpath}'' \
        'export LD_LIBRARY_PATH='${cache_ld_library_path}'' \
        '' \
        '# for Isaac Sim' \
        'unset RESOURCE_NAME' \
        '' > ${CONDA_PREFIX}/etc/conda/deactivate.d/unsetenv.sh

    # 检查是否有_isaac_sim目录 -> 如果有，表示已安装二进制文件。
    if [ -f "${isaacsim_setup_conda_env_script}" ]; then
        # 在激活期间向环境添加变量
        printf '%s\n' \
            '# for Isaac Sim' \
            'unset CARB_APP_PATH' \
            'unset EXP_PATH' \
            'unset ISAAC_PATH' \
            '' >> ${CONDA_PREFIX}/etc/conda/deactivate.d/unsetenv.sh
    fi

    # 安装一些额外的依赖项
    echo -e "[信息] 安装额外依赖项（这可能需要几分钟）..."
    conda install -c conda-forge -y importlib_metadata &> /dev/null

    # 停用环境
    conda deactivate
    # 向用户添加有关别名的信息
    echo -e "[信息] 为'isaaclab.sh'脚本向conda环境添加了'isaaclab'别名。"
    echo -e "[信息] 创建了名为'${env_name}'的conda环境。\n"
    echo -e "\t\t1. 要激活环境，运行:                conda activate ${env_name}"
    echo -e "\t\t2. 要安装Isaac Lab扩展，运行:            isaaclab -i"
    echo -e "\t\t4. 要执行格式化，运行:                      isaaclab -f"
    echo -e "\t\t5. 要停用环境，运行:              conda deactivate"
    echo -e "\n"
}

# 从模板和isaac sim设置更新vscode设置
update_vscode_settings() {
    echo "[信息] 设置vscode设置..."
    # 获取python可执行文件
    python_exe=$(extract_python_exe)
    # setup_vscode.py的路径
    setup_vscode_script="${ISAACLAB_PATH}/.vscode/tools/setup_vscode.py"
    # 在尝试运行之前检查文件是否存在
    if [ -f "${setup_vscode_script}" ]; then
        ${python_exe} "${setup_vscode_script}"
    else
        echo "[警告] 无法找到脚本'setup_vscode.py'。中止vscode设置设置。"
    fi
}

# 打印使用说明
print_help () {
    echo -e "\n用法: $(basename "$0") [-h] [-i] [-f] [-p] [-s] [-t] [-o] [-v] [-d] [-n] [-c] -- 管理Isaac Lab的实用工具。"
    echo -e "\n可选参数:"
    echo -e "\t-h, --help           显示帮助内容。"
    echo -e "\t-i, --install [LIB]  在Isaac Lab内安装扩展和学习框架作为额外依赖项。默认为'all'。"
    echo -e "\t-f, --format         运行pre-commit来格式化代码并检查lint。"
    echo -e "\t-p, --python         运行Isaac Sim或虚拟环境（如果激活）提供的python可执行文件。"
    echo -e "\t-s, --sim            运行Isaac Sim提供的模拟器可执行文件（isaac-sim.sh）。"
    echo -e "\t-t, --test           运行所有python单元测试。"
    echo -e "\t-o, --docker         运行docker容器帮助脚本（docker/container.sh）。"
    echo -e "\t-v, --vscode         从模板生成VSCode设置文件。"
    echo -e "\t-d, --docs           使用sphinx从源代码构建文档。"
    echo -e "\t-n, --new            从模板创建新的外部项目或内部任务。"
    echo -e "\t-c, --conda [NAME]   为Isaac Lab创建conda环境。默认名称为'env_isaaclab'。"
    echo -e "\n" >&2
}


#==
# 主程序
#==

# 检查提供的参数
if [ -z "$*" ]; then
    echo "[错误] 未提供参数。" >&2;
    print_help
    exit 1
fi

# 传递参数
while [[ $# -gt 0 ]]; do
    # 读取键
    case "$1" in
        -i|--install)
            # 在IsaacLab/source目录中安装python包
            echo "[信息] 在Isaac Lab仓库内安装扩展..."
            python_exe=$(extract_python_exe)
            # 递归查看目录并安装它们
            # 这不检查扩展之间的依赖关系
            export -f extract_python_exe
            export -f install_isaaclab_extension
            # 源目录
            find -L "${ISAACLAB_PATH}/source" -mindepth 1 -maxdepth 1 -type d -exec bash -c 'install_isaaclab_extension "{}"' \;
            # 为支持的强化学习框架安装python包
            echo "[信息] 安装额外要求，如学习框架..."
            # 检查是否指定了要安装的rl框架
            if [ -z "$2" ]; then
                echo "[信息] 安装所有rl框架..."
                framework_name="all"
            elif [ "$2" = "none" ]; then
                echo "[信息] 不会安装rl框架。"
                framework_name="none"
                shift # 跳过参数
            else
                echo "[信息] 安装rl框架: $2"
                framework_name=$2
                shift # 跳过参数
            fi
            # 安装指定的学习框架
            ${python_exe} -m pip install -e ${ISAACLAB_PATH}/source/isaaclab_rl["${framework_name}"]
            ${python_exe} -m pip install -e ${ISAACLAB_PATH}/source/isaaclab_mimic["${framework_name}"]

            # 检查我们是否在docker容器内或正在构建docker镜像
            # 在这种情况下不设置VSCode，因为它会要求EULA协议，这会触发用户交互
            if is_docker; then
                echo "[信息] 在docker容器内运行。跳过VSCode设置设置。"
                echo "[信息] 要设置VSCode设置，运行'isaaclab -v'。"
            else
                # 更新vscode设置
                update_vscode_settings
            fi

            # 取消设置局部变量
            unset extract_python_exe
            unset install_isaaclab_extension
            shift # 跳过参数
            ;;
        -c|--conda)
            # 如果未提供则使用默认名称
            if [ -z "$2" ]; then
                echo "[信息] 使用默认conda环境名称: env_isaaclab"
                conda_env_name="env_isaaclab"
            else
                echo "[信息] 使用conda环境名称: $2"
                conda_env_name=$2
                shift # 跳过参数
            fi
            # 为Isaac Lab设置conda环境
            setup_conda_env ${conda_env_name}
            shift # 跳过参数
            ;;
        -f|--format)
            # 重置python路径以避免与pre-commit冲突
            # 这是必要的，因为pre-commit钩子安装在单独的虚拟环境中
            # 它使用系统python来运行钩子
            if [ -n "${CONDA_DEFAULT_ENV}" ]; then
                cache_pythonpath=${PYTHONPATH}
                export PYTHONPATH=""
            fi
            # 在仓库上运行格式化器
            # 检查是否安装了pre-commit
            if ! command -v pre-commit &>/dev/null; then
                echo "[信息] 安装pre-commit..."
                pip install pre-commit
            fi
            # 始终在Isaac Lab目录内执行
            echo "[信息] 格式化仓库..."
            cd ${ISAACLAB_PATH}
            pre-commit run --all-files
            cd - > /dev/null
            # 将python路径设置回原始值
            if [ -n "${CONDA_DEFAULT_ENV}" ]; then
                export PYTHONPATH=${cache_pythonpath}
            fi
            shift # 跳过参数
            # 干净地退出
            break
            ;;
        -p|--python)
            # 运行isaacsim提供的python
            python_exe=$(extract_python_exe)
            echo "[信息] 使用来自以下位置的python: ${python_exe}"
            shift # 跳过参数
            ${python_exe} "$@"
            # 干净地退出
            break
            ;;
        -s|--sim)
            # 运行isaacsim提供的模拟器可执行文件
            isaacsim_exe=$(extract_isaacsim_exe)
            echo "[信息] 从以下位置运行isaac-sim: ${isaacsim_exe}"
            shift # 跳过参数
            ${isaacsim_exe} --ext-folder ${ISAACLAB_PATH}/source $@
            # 干净地退出
            break
            ;;
        -n|--new)
            # 运行模板生成器脚本
            python_exe=$(extract_python_exe)
            shift # 跳过参数
            echo "[信息] 安装模板依赖项..."
            ${python_exe} -m pip install -q -r ${ISAACLAB_PATH}/tools/template/requirements.txt
            echo -e "\n[信息] 运行模板生成器...\n"
            ${python_exe} ${ISAACLAB_PATH}/tools/template/cli.py $@
            # 干净地退出
            break
            ;;
        -t|--test)
            # 运行isaacsim提供的python
            python_exe=$(extract_python_exe)
            shift # 跳过参数
            ${python_exe} ${ISAACLAB_PATH}/tools/run_all_tests.py $@
            # 干净地退出
            break
            ;;
        -o|--docker)
            # 运行docker容器帮助脚本
            docker_script=${ISAACLAB_PATH}/docker/container.sh
            echo "[信息] 从以下位置运行docker实用脚本: ${docker_script}"
            shift # 跳过参数
            bash ${docker_script} $@
            # 干净地退出
            break
            ;;
        -v|--vscode)
            # 更新vscode设置
            update_vscode_settings
            shift # 跳过参数
            # 干净地退出
            break
            ;;
        -d|--docs)
            # 构建文档
            echo "[信息] 构建文档..."
            # 获取python可执行文件
            python_exe=$(extract_python_exe)
            # 安装pip包
            cd ${ISAACLAB_PATH}/docs
            ${python_exe} -m pip install -r requirements.txt > /dev/null
            # 构建文档
            ${python_exe} -m sphinx -b html -d _build/doctrees . _build/current
            # 打开文档
            echo -e "[信息] 要在默认浏览器中打开文档，运行:"
            echo -e "\n\t\txdg-open $(pwd)/_build/current/index.html\n"
            # 干净地退出
            cd - > /dev/null
            shift # 跳过参数
            # 干净地退出
            break
            ;;
        -h|--help)
            print_help
            exit 1
            ;;
        *) # 未知选项
            echo "[错误] 提供了无效参数: $1"
            print_help
            exit 1
            ;;
    esac
done

source setup_conda_env.sh