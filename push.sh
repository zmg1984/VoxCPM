#!/bin/bash
git add .
git commit -m "recover: 恢复VoxCPM部署配置"
git push origin main
echo "推送完成！请去CNB平台「构建任务」页面查看进度～"
