# =============================================================================
#  chap4 / assignment4 —— 备份 · 打包 · 恢复
#
#  默认假设：本文件与 exp_4_1_VGG19 ... exp_4_5_transformer 同级
#            （即远程的 /opt/code_chap4/）
#
#    make backup  1      # 备份 exp_4_1 的 stu_upload  -> $TESTENV/<目录名>__<时间戳>
#    make zipball 2      # 打包 exp_4_2 的 stu_upload  -> ./<目录名>-stu_upload.zip
#    make recover 3      # 用编号 3 的最新一份备份，覆盖回 exp_4_3/stu_upload
#    make recover exp_4_3_StyleTransfer_train__20261107-101010   # 恢复指定那一份
#    make list           # 列出所有备份（新的在前）
#
#  * 备份目录名 = <实验目录名>__<时间戳>，重复执行不会互相覆盖；
#  * zip 名含实验目录名，同编号重跑覆盖旧 zip，不同编号互不影响；
#  * recover 默认"整体替换"（先删后拷）；KEEP=1 改为合并拷贝；
#  * 备份根目录默认 /workspace/volume/testenv，可用 TESTENV=/path 覆盖；
#  * 实验目录的父目录默认自动识别，可用 ROOT=xxx 强制指定。
# =============================================================================

SHELL := /bin/bash
.SUFFIXES:

TESTENV ?= /workspace/volume/testenv
ROOT ?= $(if $(wildcard exp_4_1_VGG19),.,$(if $(wildcard assignment4/exp_4_1_VGG19),assignment4,.))

DIR_1 := exp_4_1_VGG19
DIR_2 := exp_4_2_StyleTransfer_infer
DIR_3 := exp_4_3_StyleTransfer_train
DIR_4 := exp_4_4_mysigmoid
DIR_5 := exp_4_5_transformer

# 编号 -> 目录名；传入的不是编号时，原样当作目录名
DIR_OF = $(if $(DIR_$(1)),$(DIR_$(1)),$(1))

# "make backup 1" 里的 1
ARG := $(word 2,$(MAKECMDGOALS))
NUM = $(if $(filter 1 2 3 4 5,$(ARG)),$(ARG),)

.PHONY: help backup zipball recover list

help:
	@echo "make backup  <1|2|3|4|5>   备份对应实验的 stu_upload 到 $(TESTENV)"
	@echo "make zipball <1|2|3|4|5>   打包对应实验的 stu_upload 为 zip（输出到当前目录）"
	@echo "make recover <1|2|3|4|5>   用该编号最新的备份恢复对应实验的 stu_upload"
	@echo "make recover <备份目录名>  恢复指定那一份备份"
	@echo "make list                  列出所有备份"
	@echo "可选：TESTENV=/path  KEEP=1(合并而非整体替换)  ROOT=<父目录>"

backup:
	@arg='$(ARG)'; if [ -z "$$arg" ]; then echo "用法: make backup <1|2|3|4|5>"; exit 1; fi; \
	set -e; dir='$(call DIR_OF,$(ARG))'; src='$(ROOT)'/"$$dir"/stu_upload; \
	[ -d "$$src" ] || { echo "错误: 目录不存在 -> $$src"; exit 1; }; \
	mkdir -p '$(TESTENV)'; \
	ts=$$(date +%Y%m%d-%H%M%S); \
	dst='$(TESTENV)'/"$$dir"__$$ts; n=1; \
	while [ -e "$$dst" ]; do dst='$(TESTENV)'/"$$dir"__$${ts}_$$n; n=$$((n+1)); done; \
	cp -a "$$src" "$$dst"; \
	printf 'source: %s\nhost  : %s\ntime  : %s\n' "$$(cd "$$src" && pwd)" "$$(hostname)" "$$(date '+%F %T')" > "$$dst.info"; \
	echo "备份完成: $$dst"

zipball:
	@arg='$(ARG)'; if [ -z "$$arg" ]; then echo "用法: make zipball <1|2|3|4|5>"; exit 1; fi; \
	set -e; dir='$(call DIR_OF,$(ARG))'; src='$(ROOT)'/"$$dir"/stu_upload; \
	[ -d "$$src" ] || { echo "错误: 目录不存在 -> $$src"; exit 1; }; \
	command -v zip >/dev/null 2>&1 || { echo "错误: 未安装 zip"; exit 1; }; \
	out="$$(pwd)/$$dir-stu_upload.zip"; rm -f "$$out"; \
	( cd "$$src" && zip -r -q "$$out" . \
	    -x '*/__pycache__/*' '__pycache__/*' '*.pyc' '*/build/*' 'build/*' ); \
	echo "打包完成: $$out"

recover:
	@arg='$(ARG)'; if [ -z "$$arg" ]; then echo "用法: make recover <1|2|3|4|5|备份目录名>"; exit 1; fi; \
	set -e; \
	if [ -n '$(NUM)' ]; then \
	    dir='$(DIR_$(ARG))'; \
	elif [ -d '$(ROOT)'/"$$arg"/stu_upload ]; then \
	    dir="$$arg"; \
	else \
	    src='$(TESTENV)'/"$$arg"; dir=$${arg%%__*}; \
	fi; \
	if [ -z "$$src" ]; then \
	    src=$$(ls -1dt '$(TESTENV)'/"$$dir"__*/ 2>/dev/null | head -n1); src=$${src%/}; \
	fi; \
	[ -n "$$src" ] && [ -d "$$src" ] || { echo "错误: 找不到备份 -> $${src:-$(TESTENV)/$$arg}"; exit 1; }; \
	target='$(ROOT)'/"$$dir"/stu_upload; mkdir -p "$$target"; \
	if [ "$${KEEP:-0}" = "1" ]; then cp -a "$$src"/. "$$target"/; \
	else rm -rf "$$target"; cp -a "$$src" "$$target"; fi; \
	echo "已恢复: $$src -> $$target"

list:
	@if ls -1dt '$(TESTENV)'/*__*/ >/dev/null 2>&1; then \
	    ls -1dt '$(TESTENV)'/*__*/ | sed 's#/$$##'; \
	else echo "(还没有备份)"; fi

# 允许 "make backup 1" 里的 1/2/3/4/5（以及任意附加词）被当成无副作用的空目标
%:
	@:
