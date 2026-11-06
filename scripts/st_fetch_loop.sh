#!/bin/bash
# ST 标志采集循环：每 25 分钟重启一次进程（保持新连接的高速阶段），直到 666 天全部完成
cd /Users/taofeng/Documents/deepseek-harness/default-workspace/Sequoia-X || exit 1
for i in $(seq 1 60); do
  DONE=$(sqlite3 data/sequoia_v2.db "SELECT COUNT(*) FROM st_meta;" 2>/dev/null)
  if [ "${DONE:-0}" -ge 666 ]; then
    echo "ALL DONE: $DONE days"
    break
  fi
  echo "=== 第 $i 轮启动，已完成 $DONE 天 ==="
  .venv/bin/python scripts/fetch_st_flags.py &
  PID=$!
  sleep 1500
  kill $PID 2>/dev/null
  wait $PID 2>/dev/null
  sleep 5
done
echo "LOOP EXIT"
