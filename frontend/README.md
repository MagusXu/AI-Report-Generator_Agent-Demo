# 前端

本地 AI 报告生成 Agent Demo 的 React + TypeScript + Vite 前端。

项目总览见仓库根目录 [README.md](../README.md)。

## 职责范围

- 报告项目配置与工作台界面
- 参考资料上传 / 勾选 / 入库状态
- 章节 Prompt、AI 辅助起草与 RAG 生成控制
- SSE 流式预览与执行阶段提示
- 表格设置：AI 汇总 / 引用原表（标题与检索描述分离）
- 正文渲染：标题、加粗、引用、表格及表下灰色小字表题
- 版本管理、确认、整份预览与模拟导出检查

## 本地命令

Node 22：

```bash
npm install
npm run dev
```

本地地址：`http://127.0.0.1:5173`

```bash
npm run build
```
