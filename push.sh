#!/bin/bash
# 一键提交代码并触发镜像构建
echo "开始提交代码..."
git add .
git commit -m "feat: 初始化VoxCPM部署配置"
git push origin main
echo "推送完成！请去CNB平台「构建任务」页面查看进度～"
